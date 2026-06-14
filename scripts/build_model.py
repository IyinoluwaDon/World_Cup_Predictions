import os, random, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
from scipy.stats import poisson as sp_poisson

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

# Models
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, log_loss, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
print("All imports OK.")






df = pd.read_csv(os.path.join(DATA_DIR, "processed", "master_matches.csv"))
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
df.head()






print("Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nResult distribution:\n{df['result'].value_counts()}")
print(f"\nTop tournaments:\n{df['tournament'].value_counts().head(8)}")




fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Match Distributions", fontsize=14, fontweight="bold")

result_labels = {1:"Home Win", 0:"Draw", -1:"Away Win"}
counts = df["result"].map(result_labels).value_counts()
axes[0].bar(counts.index, counts.values, color=["#2ecc71","#95a5a6","#e74c3c"], edgecolor="white")
axes[0].set_title("Outcome Distribution")
for bar, val in zip(axes[0].patches, counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                 f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=9)

axes[1].hist(df["home_score"], bins=range(0,10), alpha=0.7, label="Home", color="#3498db", edgecolor="white")
axes[1].hist(df["away_score"], bins=range(0,10), alpha=0.7, label="Away", color="#e67e22", edgecolor="white")
axes[1].set_title("Goals Distribution"); axes[1].legend()

axes[2].hist(df["home_ranking_pts"].dropna(), bins=40, color="#9b59b6", edgecolor="white", alpha=0.8)
axes[2].set_title("FIFA Ranking Points Distribution")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "eda_distributions.png"), dpi=150, bbox_inches="tight")
plt.show()





TOURNAMENT_WEIGHTS = {
    'FIFA World Cup':60, 'UEFA Euro':50, 'Copa America':50,
    'African Cup of Nations':50, 'AFC Asian Cup':50, 'Gold Cup':40,
    'CONCACAF Nations League':35, 'UEFA Nations League':35,
    'UEFA Euro qualification':30, 'FIFA World Cup qualification':30,
    'African Cup of Nations qualification':25, 'AFC Asian Cup qualification':25,
}

def get_k(tournament):
    for key, k in TOURNAMENT_WEIGHTS.items():
        if key.lower() in tournament.lower(): return k
    return 25

# Compute Elo chronologically — record BEFORE each match to avoid leakage
elo = defaultdict(lambda: 1500.0)
home_elo_w, away_elo_w = [], []

for _, row in df.iterrows():
    home, away = row["home_team"], row["away_team"]
    he, ae = elo[home], elo[away]
    home_elo_w.append(he); away_elo_w.append(ae)
    exp_h = 1 / (1 + 10**((ae - he) / 400))
    sh = 1.0 if row["result"] == 1 else (0.5 if row["result"] == 0 else 0.0)
    k = get_k(row["tournament"])
    elo[home] = he + k * (sh - exp_h)
    elo[away] = ae + k * ((1-sh) - (1-exp_h))

df["home_elo_w"] = home_elo_w
df["away_elo_w"] = away_elo_w
df["elo_diff_w"] = df["home_elo_w"] - df["away_elo_w"]
elo_state_final  = dict(elo)

print("Top 10 teams by final Elo rating:")
for rank, (team, rating) in enumerate(sorted(elo_state_final.items(), key=lambda x:-x[1])[:10], 1):
    print(f"  {rank:>2}. {team:<25} {rating:.0f}")






home_form_5, away_form_5 = [], []

for _, row in df.iterrows():
    date, ht, at = row["date"], row["home_team"], row["away_team"]
    hh = df[(df["home_team"]==ht) & (df["date"]<date)]
    ha = df[(df["away_team"]==ht) & (df["date"]<date)]
    ah = df[(df["home_team"]==at) & (df["date"]<date)]
    aa = df[(df["away_team"]==at) & (df["date"]<date)]

    hw5 = list(hh.tail(5)["result"].map(lambda x:1 if x==1 else 0))
    aw5 = list(ha.tail(5)["result"].map(lambda x:1 if x==-1 else 0))
    all5_h = sorted(zip(list(hh.tail(5)["date"])+list(ha.tail(5)["date"]), hw5+aw5))[-5:]
    home_form_5.append(np.mean([w for _,w in all5_h]) if all5_h else 0.5)

    hw5a = list(ah.tail(5)["result"].map(lambda x:1 if x==1 else 0))
    aw5a = list(aa.tail(5)["result"].map(lambda x:1 if x==-1 else 0))
    all5_a = sorted(zip(list(ah.tail(5)["date"])+list(aa.tail(5)["date"]), hw5a+aw5a))[-5:]
    away_form_5.append(np.mean([w for _,w in all5_a]) if all5_a else 0.5)

df["home_form_5"] = home_form_5
df["away_form_5"] = away_form_5
df["form_diff"]   = df["home_form_5"] - df["away_form_5"]
print("Form-5 done.")
print(df[["home_form_5","away_form_5","form_diff"]].describe().round(3))







home_f10, away_f10 = [], []
hgs5, hgc5, ags5, agc5 = [], [], [], []

for i, (idx, row) in enumerate(df.iterrows()):
    if i % 3000 == 0: print(f"  {i}/{len(df)}...")
    date, ht, at = row["date"], row["home_team"], row["away_team"]
    hh = df[(df["home_team"]==ht) & (df["date"]<date)]
    ha = df[(df["away_team"]==ht) & (df["date"]<date)]
    ah = df[(df["home_team"]==at) & (df["date"]<date)]
    aa = df[(df["away_team"]==at) & (df["date"]<date)]

    def form10(hm, am):
        hw = hm[["date"]].assign(w=hm["result"].map(lambda x:1 if x==1 else 0))
        aw = am[["date"]].assign(w=am["result"].map(lambda x:1 if x==-1 else 0))
        c  = pd.concat([hw, aw]).sort_values("date").tail(10)
        return c["w"].mean() if len(c) else 0.5

    home_f10.append(form10(hh, ha)); away_f10.append(form10(ah, aa))

    hh5=hh.tail(5); ha5=ha.tail(5); ah5=ah.tail(5); aa5=aa.tail(5)
    gs_h=list(hh5["home_score"])+list(ha5["away_score"])
    gc_h=list(hh5["away_score"])+list(ha5["home_score"])
    gs_a=list(ah5["home_score"])+list(aa5["away_score"])
    gc_a=list(ah5["away_score"])+list(aa5["home_score"])
    hgs5.append(np.mean(gs_h) if gs_h else 1.2); hgc5.append(np.mean(gc_h) if gc_h else 1.0)
    ags5.append(np.mean(gs_a) if gs_a else 1.2); agc5.append(np.mean(gc_a) if gc_a else 1.0)

df["home_form_10"]=home_f10; df["away_form_10"]=away_f10
df["form_diff_10"]=df["home_form_10"]-df["away_form_10"]
df["home_goals_scored_5"]=hgs5; df["home_goals_conceded_5"]=hgc5
df["away_goals_scored_5"]=ags5; df["away_goals_conceded_5"]=agc5
df["goal_diff_scored"]=df["home_goals_scored_5"]-df["away_goals_scored_5"]
df["goal_diff_conceded"]=df["away_goals_conceded_5"]-df["home_goals_conceded_5"]
print("\nForm-10 + Goals done.")






def get_tournament_importance(t):
    t = t.lower()
    if "world cup" in t and "qualif" not in t: return 3
    if any(x in t for x in ["euro","copa","cup of nations","asian cup","gold cup"]) and "qualif" not in t: return 2
    if "nations league" in t or "qualif" in t: return 1
    return 0

h2h_wins, h2h_draws, h2h_losses = [], [], []
for _, row in df.iterrows():
    past = df[(df["home_team"]==row["home_team"]) &
              (df["away_team"]==row["away_team"]) &
              (df["date"]<row["date"])]
    h2h_wins.append(int((past["result"]==1).sum()))
    h2h_draws.append(int((past["result"]==0).sum()))
    h2h_losses.append(int((past["result"]==-1).sum()))

df["h2h_home_wins"]  = h2h_wins
df["h2h_draws"]      = h2h_draws
df["h2h_home_losses"]= h2h_losses
df["ranking_diff"]   = df["home_ranking_pts"] - df["away_ranking_pts"]
df["is_neutral"]     = df["neutral"].astype(int)
df["same_confederation"] = (df["home_confederation"]==df["away_confederation"]).astype(int)
df["tournament_importance"] = df["tournament"].apply(get_tournament_importance)
print("Context features done.")








FEATURES = [
    "elo_diff_w",              # Weighted Elo gap
    "ranking_diff",            # FIFA points gap
    "home_form_5",             # Short-term home form
    "away_form_5",             # Short-term away form
    "form_diff",               # Short-term form gap
    "home_form_10",            # Medium-term home form
    "away_form_10",            # Medium-term away form
    "form_diff_10",            # Medium-term form gap
    "home_goals_scored_5",     # Home attack quality
    "home_goals_conceded_5",   # Home defensive solidity
    "away_goals_scored_5",     # Away attack quality
    "away_goals_conceded_5",   # Away defensive solidity
    "goal_diff_scored",        # Relative attacking gap
    "goal_diff_conceded",      # Relative defensive gap
    "h2h_home_wins",           # Head-to-head history
    "h2h_draws",
    "h2h_home_losses",
    "is_neutral",              # Venue context
    "same_confederation",      # Confederation context
    "tournament_importance",   # Match stakes
]

POISSON_FEATURES = [
    "elo_diff_w","ranking_diff","home_form_5","away_form_5","form_diff",
    "home_goals_scored_5","home_goals_conceded_5",
    "away_goals_scored_5","away_goals_conceded_5",
    "goal_diff_scored","goal_diff_conceded",
    "is_neutral","same_confederation","tournament_importance",
]

le = LabelEncoder()
df["result_encoded"] = le.fit_transform(df["result"])
print(f"Total features: {len(FEATURES)}")
print(f"Label classes: {le.classes_}  →  encoded as {le.transform(le.classes_)}")






train_df = df[df["date"] < "2018-01-01"]
cal_df   = df[(df["date"] >= "2018-01-01") & (df["date"] < "2022-01-01")]
test_df  = df[df["date"] >= "2022-01-01"]

X_train = train_df[FEATURES].fillna(0); y_train = train_df["result_encoded"]
X_cal   = cal_df[FEATURES].fillna(0);   y_cal   = cal_df["result_encoded"]
X_test  = test_df[FEATURES].fillna(0);  y_test  = test_df["result_encoded"]

print(f"Train : {len(X_train):,} matches (before 2018)")
print(f"Cal   : {len(X_cal):,} matches (2018–2021)")
print(f"Test  : {len(X_test):,} matches (2022 onward)")







tscv = TimeSeriesSplit(n_splits=5)

lr_grid = GridSearchCV(
    LogisticRegression(solver="lbfgs", max_iter=2000),
    param_grid={"C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
    cv=tscv, scoring="neg_log_loss", n_jobs=-1,
)
lr_grid.fit(X_train, y_train)
best_lr = lr_grid.best_estimator_
print(f"Best C: {lr_grid.best_params_['C']}")
print(f"Best CV log-loss: {-lr_grid.best_score_:.4f}")







print("XGBoost...")
xgb_m = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
xgb_m.fit(X_train, y_train)

print("LightGBM...")
lgb_m = LGBMClassifier(objective="multiclass", num_class=3, metric="multi_logloss",
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
lgb_m.fit(X_train, y_train)

print("Random Forest...")
rf_m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_m.fit(X_train, y_train)
print("All models trained.")






# Isotonic calibration corrects systematic probability biases.
# Fitted on 2018-2021 holdout — separate from training and test sets.
print("Calibrating on 2018–2021 holdout...")
lr_c  = CalibratedClassifierCV(best_lr, method="isotonic", cv=None, ensemble=False); lr_c.fit(X_cal, y_cal)
xgb_c = CalibratedClassifierCV(xgb_m,  method="isotonic", cv=None, ensemble=False); xgb_c.fit(X_cal, y_cal)
lgb_c = CalibratedClassifierCV(lgb_m,  method="isotonic", cv=None, ensemble=False); lgb_c.fit(X_cal, y_cal)
rf_c  = CalibratedClassifierCV(rf_m,   method="isotonic", cv=None, ensemble=False); rf_c.fit(X_cal, y_cal)
print("Done.")






print(f"{'Model':<35} {'Accuracy':>10} {'Log-Loss':>10}")
print("-"*57)
model_store = {}
for name, m in [("LR (original)",best_lr),("LR (calibrated)",lr_c),
                ("XGBoost (calibrated)",xgb_c),("LightGBM (calibrated)",lgb_c),
                ("Random Forest (calibrated)",rf_c)]:
    probs=m.predict_proba(X_test); preds=m.predict(X_test)
    ll=log_loss(y_test,probs); acc=accuracy_score(y_test,preds)
    print(f"  {name:<33} {acc:>10.4f} {ll:>10.4f}")
    model_store[name]={"model":m,"ll":ll,"acc":acc,"preds":preds,"probs":probs}

# Soft-vote ensemble
ens_probs=np.mean([lr_c.predict_proba(X_test),xgb_c.predict_proba(X_test),
                   lgb_c.predict_proba(X_test),rf_c.predict_proba(X_test)],axis=0)
ens_preds=np.argmax(ens_probs,axis=1)
ens_ll=log_loss(y_test,ens_probs); ens_acc=accuracy_score(y_test,ens_preds)
print(f"  {'Soft-Vote Ensemble':<33} {ens_acc:>10.4f} {ens_ll:>10.4f}")

best_name=min(model_store,key=lambda k:model_store[k]["ll"])
best_model=model_store[best_name]["model"]
print(f"\n✅ Best model: {best_name} (ll={model_store[best_name]['ll']:.4f})")






class_names=["Away Win","Draw","Home Win"]
fig,axes=plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Confusion Matrix — Best Model vs Ensemble",fontsize=13,fontweight="bold")
for ax,(title,preds) in zip(axes,[(f"Best: {best_name}",model_store[best_name]["preds"]),("Soft-Vote Ensemble",ens_preds)]):
    cm=confusion_matrix(y_test,preds)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
    disp.plot(ax=ax,colorbar=False,cmap="Blues"); ax.set_title(title,fontsize=11)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            pct=cm[i,j]/cm[i].sum()*100
            ax.text(j,i+0.35,f"({pct:.0f}%)",ha="center",va="center",fontsize=8,
                    color="white" if cm[i,j]>cm.max()/2 else "gray")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"),dpi=150,bbox_inches="tight")
plt.show()
print("Key: Draw row will be mostly off-diagonal — draws are the hardest class to predict.")






print("=== Best Model ===")
print(classification_report(y_test, model_store[best_name]["preds"], target_names=class_names))
print("=== Ensemble ===")
print(classification_report(y_test, ens_preds, target_names=class_names))





coefficients = best_lr.coef_[2]  # Home Win class
feat_imp = pd.DataFrame({"Feature":FEATURES,"Coefficient":coefficients}).sort_values("Coefficient",key=abs,ascending=False)
fig,ax=plt.subplots(figsize=(9,7))
colors=["#2ecc71" if c>0 else "#e74c3c" for c in feat_imp["Coefficient"]]
ax.barh(feat_imp["Feature"],feat_imp["Coefficient"],color=colors,edgecolor="white",height=0.6)
ax.axvline(0,color="white",linewidth=0.8,alpha=0.5)
ax.set_xlabel("Coefficient (Home Win class)")
ax.set_title("Feature Importance — LR Coefficients\nGreen=pushes toward Home Win | Red=pushes away",fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"),dpi=150,bbox_inches="tight")
plt.show()
print("\nTop 5:")
print(feat_imp.head(5).to_string(index=False))






X_ptr = train_df[POISSON_FEATURES].fillna(0)
X_pte = test_df[POISSON_FEATURES].fillna(0)
y_htr = train_df["home_score"].clip(0,8)
y_atr = train_df["away_score"].clip(0,8)
y_hte = test_df["home_score"].clip(0,8)
y_ate = test_df["away_score"].clip(0,8)

print("Fitting Poisson home goals model...")
poisson_home = PoissonRegressor(alpha=0.1, max_iter=1000)
poisson_home.fit(X_ptr, y_htr)

print("Fitting Poisson away goals model...")
poisson_away = PoissonRegressor(alpha=0.1, max_iter=1000)
poisson_away.fit(X_ptr, y_atr)

hp = poisson_home.predict(X_pte)
ap = poisson_away.predict(X_pte)
print(f"\nHome goals MAE : {mean_absolute_error(y_hte, hp):.3f}")
print(f"Away goals MAE : {mean_absolute_error(y_ate, ap):.3f}")
print(f"Avg pred home  : {hp.mean():.2f}  (actual: {y_hte.mean():.2f})")
print(f"Avg pred away  : {ap.mean():.2f}  (actual: {y_ate.mean():.2f})")







# Validate Poisson outcome probabilities
def poisson_match_probs(lh, la, mg=8):
    p_hw=p_d=p_aw=0.0
    for h in range(mg+1):
        for a in range(mg+1):
            p=sp_poisson.pmf(h,lh)*sp_poisson.pmf(a,la)
            if h>a: p_hw+=p
            elif h==a: p_d+=p
            else: p_aw+=p
    t=p_hw+p_d+p_aw
    return p_hw/t,p_d/t,p_aw/t

pois_preds=[]
for i in range(len(X_pte)):
    pw,pd_,paw=poisson_match_probs(max(0.1,hp[i]),max(0.1,ap[i]))
    pois_preds.append([pw,pd_,paw])
pois_arr=np.array(pois_preds)

classes=list(le.classes_)
idx_hw=classes.index(1); idx_d=classes.index(0); idx_aw=classes.index(-1)
pois_reordered=np.zeros_like(pois_arr)
pois_reordered[:,idx_hw]=pois_arr[:,0]
pois_reordered[:,idx_d] =pois_arr[:,1]
pois_reordered[:,idx_aw]=pois_arr[:,2]

pois_ll=log_loss(y_test,pois_reordered)
pois_acc=accuracy_score(y_test,np.argmax(pois_reordered,axis=1))
print(f"Poisson outcome log-loss : {pois_ll:.4f} | accuracy: {pois_acc:.4f}")
print(f"LR (best) log-loss       : {model_store[best_name]['ll']:.4f}")
print(f"\nConclusion: LR has better log-loss → used for win/draw/loss predictions.")
print("Poisson is used for xG scoreline display only.")







WC2026_GROUPS = {
    "A":["Mexico","South Africa","Korea Republic","New Zealand"],
    "B":["Canada","Switzerland","Qatar","Honduras"],
    "C":["Brazil","Morocco","Haiti","Scotland"],
    "D":["USA","Paraguay","Australia","Cameroon"],
    "E":["Germany","Curacao","Ivory Coast","Ecuador"],
    "F":["Netherlands","Japan","Tunisia","Senegal"],
    "G":["Belgium","Egypt","Iran","Costa Rica"],
    "H":["Spain","Cape Verde","Saudi Arabia","Uruguay"],
    "I":["France","Algeria","Norway","Costa Rica"],
    "J":["Argentina","Algeria","Austria","Jordan"],
    "K":["Portugal","Colombia","Uzbekistan","Thailand"],
    "L":["England","Croatia","Ghana","Panama"],
}
ALL_TEAMS = list({t for grp in WC2026_GROUPS.values() for t in grp})
HOST_NATIONS = {"USA","Canada","Mexico"}
CONFEDERATION_MAP = {
    "Spain":"UEFA","France":"UEFA","England":"UEFA","Germany":"UEFA","Netherlands":"UEFA",
    "Belgium":"UEFA","Portugal":"UEFA","Croatia":"UEFA","Switzerland":"UEFA","Norway":"UEFA",
    "Scotland":"UEFA","Austria":"UEFA","Brazil":"CONMEBOL","Argentina":"CONMEBOL",
    "Colombia":"CONMEBOL","Uruguay":"CONMEBOL","Ecuador":"CONMEBOL","Paraguay":"CONMEBOL",
    "Morocco":"CAF","Senegal":"CAF","Egypt":"CAF","Ivory Coast":"CAF","Ghana":"CAF",
    "Cape Verde":"CAF","South Africa":"CAF","Tunisia":"CAF","Algeria":"CAF","Cameroon":"CAF",
    "Japan":"AFC","Korea Republic":"AFC","Iran":"AFC","Saudi Arabia":"AFC","Australia":"AFC",
    "Uzbekistan":"AFC","Jordan":"AFC","Qatar":"AFC","Thailand":"AFC",
    "Mexico":"CONCACAF","USA":"CONCACAF","Canada":"CONCACAF","Panama":"CONCACAF",
    "Curacao":"CONCACAF","Haiti":"CONCACAF","Honduras":"CONCACAF","Costa Rica":"CONCACAF",
    "New Zealand":"OFC",
}
print(f"Teams: {len(ALL_TEAMS)}")






def build_team_stats(df, elo_state):
    stats = {}
    for team in ALL_TEAMS:
        hm=df[df["home_team"]==team]; am=df[df["away_team"]==team]
        hm5=hm.tail(5); am5=am.tail(5)
        hw5=list(hm5["result"].map(lambda x:1 if x==1 else 0))
        aw5=list(am5["result"].map(lambda x:1 if x==-1 else 0))
        all5=sorted(zip(list(hm5["date"])+list(am5["date"]),hw5+aw5))[-5:]
        form5=np.mean([w for _,w in all5]) if all5 else 0.45
        hw10=hm[["date"]].assign(w=hm["result"].map(lambda x:1 if x==1 else 0))
        aw10=am[["date"]].assign(w=am["result"].map(lambda x:1 if x==-1 else 0))
        all10=pd.concat([hw10,aw10]).sort_values("date").tail(10)
        form10=all10["w"].mean() if len(all10) else 0.45
        gs=list(hm5["home_score"])+list(am5["away_score"])
        gc=list(hm5["away_score"])+list(am5["home_score"])
        pts=1200.0
        if len(hm): pts=float(hm.iloc[-1]["home_ranking_pts"])
        elif len(am): pts=float(am.iloc[-1]["away_ranking_pts"])
        stats[team]={"elo":elo_state.get(team,1500.0),"pts":pts,"form5":form5,
            "form10":form10,"goals_scored5":np.mean(gs) if gs else 1.2,
            "goals_conceded5":np.mean(gc) if gc else 1.0,
            "confederation":CONFEDERATION_MAP.get(team,"UEFA")}
    return stats

team_stats = build_team_stats(df, elo_state_final)
print("Team stats built. Sample — France:", team_stats.get("France",{}))







# Batch precompute all matchup probabilities — avoids calling predict_proba
# inside 10,000 simulation loops (would be very slow).
print("Precomputing matchup probabilities...")
rows, pairs = [], []
for h in ALL_TEAMS:
    for a in ALL_TEAMS:
        if h==a: continue
        hs=team_stats[h]; as_=team_stats[a]
        same_c=1 if hs["confederation"]==as_["confederation"] else 0
        is_n=0 if h in HOST_NATIONS else 1
        rows.append({"elo_diff_w":hs["elo"]-as_["elo"],"ranking_diff":hs["pts"]-as_["pts"],
            "home_form_5":hs["form5"],"away_form_5":as_["form5"],"form_diff":hs["form5"]-as_["form5"],
            "home_form_10":hs["form10"],"away_form_10":as_["form10"],"form_diff_10":hs["form10"]-as_["form10"],
            "home_goals_scored_5":hs["goals_scored5"],"home_goals_conceded_5":hs["goals_conceded5"],
            "away_goals_scored_5":as_["goals_scored5"],"away_goals_conceded_5":as_["goals_conceded5"],
            "goal_diff_scored":hs["goals_scored5"]-as_["goals_scored5"],
            "goal_diff_conceded":as_["goals_conceded5"]-hs["goals_conceded5"],
            "h2h_home_wins":0,"h2h_draws":0,"h2h_home_losses":0,
            "is_neutral":is_n,"same_confederation":same_c,"tournament_importance":3})
        pairs.append((h,a))
batch=best_model.predict_proba(pd.DataFrame(rows)[FEATURES]).astype(float)
batch/=batch.sum(axis=1,keepdims=True)
ihw=classes.index(1); id_=classes.index(0); iaw=classes.index(-1)
MATCHUPS={(h,a):(batch[i][ihw],batch[i][id_],batch[i][iaw]) for i,(h,a) in enumerate(pairs)}
print(f"Precomputed {len(MATCHUPS):,} matchups.")






def sim_group(teams):
    pts={t:0 for t in teams}; gd={t:0 for t in teams}
    gf={t:0 for t in teams}; h2h_pts=defaultdict(int); h2h_gd=defaultdict(int)
    for i in range(len(teams)):
        for j in range(i+1,len(teams)):
            h,a=teams[i],teams[j]; hw,d,aw=MATCHUPS.get((h,a),(0.4,0.2,0.4))
            r=random.random()
            if r<hw: hs_=random.randint(1,3); as__=random.randint(0,max(0,hs_-1)); pts[h]+=3; h2h_pts[h]+=3
            elif r<hw+d: hs_=random.randint(0,2); as__=hs_; pts[h]+=1; pts[a]+=1; h2h_pts[h]+=1; h2h_pts[a]+=1
            else: as__=random.randint(1,3); hs_=random.randint(0,max(0,as__-1)); pts[a]+=3; h2h_pts[a]+=3
            diff=hs_-as__; gd[h]+=diff; gd[a]-=diff; gf[h]+=hs_; gf[a]+=as__
            h2h_gd[h]+=diff; h2h_gd[a]-=diff
    ranked=sorted(teams,key=lambda t:(pts[t],h2h_pts[t],h2h_gd[t],gd[t],gf[t],random.random()),reverse=True)
    return ranked,pts,gd,gf

def ko_winner(h,a):
    hw,d,aw=MATCHUPS.get((h,a),(0.4,0.2,0.4)); p_h=hw+d/2; p_a=aw+d/2
    pr=np.array([p_h,p_a]); pr/=pr.sum()
    return h if random.random()<pr[0] else a

N_SIMS=10_000; wins=defaultdict(int); sf_c=defaultdict(int); f_c=defaultdict(int)
print(f"Running {N_SIMS:,} simulations...")
for _ in range(N_SIMS):
    qualifiers=[]; thirds=[]
    for _,teams in WC2026_GROUPS.items():
        ranked,pts,gd,gf=sim_group(teams)
        qualifiers.append(ranked[0]); qualifiers.append(ranked[1])
        thirds.append((ranked[2],pts[ranked[2]],gd[ranked[2]],gf[ranked[2]]))
    thirds.sort(key=lambda x:(-x[1],-x[2],-x[3],random.random()))
    qualifiers+=[x[0] for x in thirds[:8]]
    random.shuffle(qualifiers); bracket=list(qualifiers); rnd=0
    while len(bracket)>1:
        if rnd==3:
            for t in bracket: sf_c[t]+=1
        if rnd==4:
            for t in bracket: f_c[t]+=1
        bracket=[ko_winner(bracket[i],bracket[i+1]) for i in range(0,len(bracket)-1,2)]
        rnd+=1
    wins[bracket[0]]+=1

print(f"\n{'Rank':<5}{'Team':<25}{'Win%':>7}{'Final%':>8}{'Semi%':>8}")
print("-"*50)
for rank,(team,cnt) in enumerate(sorted(wins.items(),key=lambda x:-x[1])[:15],1):
    print(f"{rank:<5}{team:<25}{cnt/N_SIMS*100:>6.2f}%{f_c[team]/N_SIMS*100:>7.2f}%{sf_c[team]/N_SIMS*100:>7.2f}%")

sim_results={team:{"win_pct":wins[team]/N_SIMS*100,
                    "final_pct":f_c[team]/N_SIMS*100,
                    "semi_pct":sf_c[team]/N_SIMS*100} for team in ALL_TEAMS}






# This file is what the Streamlit app loads.
# It contains everything the app needs to run without re-running this notebook.

use_ensemble = (ens_ll < model_store[best_name]["ll"])
print(f"Best model: {best_name}  |  Use ensemble: {use_ensemble}")

with open(os.path.join(MODELS_DIR, "best_model.pkl"),"wb") as f:
    pickle.dump({
        # Primary prediction model (lowest log-loss)
        "model":           best_model,
        # All calibrated models (for ensemble option)
        "models_ensemble": [lr_c, xgb_c, lgb_c, rf_c],
        "use_ensemble":    use_ensemble,
        # Poisson xG models
        "poisson_home":    poisson_home,
        "poisson_away":    poisson_away,
        "poisson_feats":   POISSON_FEATURES,
        # Encoding and features
        "le":              le,
        "features":        FEATURES,
        "idx_hw":          idx_hw,
        "idx_d":           idx_d,
        "idx_aw":          idx_aw,
        # Data
        "elo_state":       elo_state_final,
        "df":              df,
        # Results
        "sim_results":     sim_results,
        "model_results": {
            "LR_original":    {"acc":model_store["LR (original)"]["acc"],             "ll":model_store["LR (original)"]["ll"]},
            "LR_calibrated":  {"acc":model_store["LR (calibrated)"]["acc"],           "ll":model_store["LR (calibrated)"]["ll"]},
            "XGB":            {"acc":model_store["XGBoost (calibrated)"]["acc"],       "ll":model_store["XGBoost (calibrated)"]["ll"]},
            "LGB":            {"acc":model_store["LightGBM (calibrated)"]["acc"],      "ll":model_store["LightGBM (calibrated)"]["ll"]},
            "RF":             {"acc":model_store["Random Forest (calibrated)"]["acc"], "ll":model_store["Random Forest (calibrated)"]["ll"]},
            "Ensemble":       {"acc":ens_acc,  "ll":ens_ll},
            "Poisson":        {"acc":pois_acc, "ll":pois_ll},
        },
    }, f)

import os
size_mb = os.path.getsize(os.path.join(MODELS_DIR, "best_model.pkl"))/1e6
print(f"\n✅ Saved → models/best_model.pkl ({size_mb:.1f} MB)")
print("The Streamlit app (wc2026_app.py) loads this file directly.")






CONF_COLORS={"UEFA":"#1a78cf","CONMEBOL":"#2ecc71","CAF":"#e67e22",
             "AFC":"#e74c3c","CONCACAF":"#27ae60","OFC":"#95a5a6"}

# Chart 1: Win probabilities — all teams
all_sorted=sorted(ALL_TEAMS,key=lambda t:-wins[t])
probs_s=[wins[t]/N_SIMS*100 for t in all_sorted]
cols_s=[CONF_COLORS.get(CONFEDERATION_MAP.get(t,"?"),"#888") for t in all_sorted]

fig,ax=plt.subplots(figsize=(11,max(8,len(all_sorted)*0.38)))
bars=ax.barh(all_sorted,probs_s,color=cols_s,edgecolor="white",linewidth=0.4)
ax.axvline(100/48,color="gold",linestyle="--",linewidth=1.2,alpha=0.8,label=f"Random ({100/48:.1f}%)")
for bar,val in zip(bars,probs_s):
    if val>=0.4: ax.text(val+0.1,bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va="center",fontsize=8,color="white")
ax.set_xlabel("Win Probability (%)")
ax.set_title(f"WC 2026 Win Probabilities — {N_SIMS:,} Monte Carlo Simulations",fontsize=13,fontweight="bold")
patches=[mpatches.Patch(color=v,label=k) for k,v in CONF_COLORS.items()]
ax.legend(handles=patches+[ax.lines[0]],loc="lower right",fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "win_probabilities.png"),dpi=150,bbox_inches="tight")
plt.show()
print("Saved → outputs/win_probabilities.png")



# Chart 2: Top 10 — Win / Final / Semi
top10=sorted(ALL_TEAMS,key=lambda t:-wins[t])[:10]
x=np.arange(len(top10)); w=0.25
fig,ax=plt.subplots(figsize=(12,5))
ax.bar(x-w,[wins[t]/N_SIMS*100 for t in top10],w,label="Win %",color="#c8a84b",edgecolor="white")
ax.bar(x,  [f_c[t]/N_SIMS*100 for t in top10], w,label="Final %",color="#3498db",edgecolor="white")
ax.bar(x+w,[sf_c[t]/N_SIMS*100 for t in top10],w,label="Semi %",color="#2ecc71",edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(top10,rotation=30,ha="right")
ax.set_ylabel("Probability (%)"); ax.set_title("Top 10 — Win / Final / Semi %",fontsize=12,fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "top10_probabilities.png"),dpi=150,bbox_inches="tight")
plt.show()





# Chart 3: Model comparison
names=["LR (original)","LR (calibrated)","XGBoost","LightGBM","Random Forest","Ensemble","Poisson"]
lls=[model_store["LR (original)"]["ll"],model_store["LR (calibrated)"]["ll"],
     model_store["XGBoost (calibrated)"]["ll"],model_store["LightGBM (calibrated)"]["ll"],
     model_store["Random Forest (calibrated)"]["ll"],ens_ll,pois_ll]
accs=[model_store["LR (original)"]["acc"],model_store["LR (calibrated)"]["acc"],
      model_store["XGBoost (calibrated)"]["acc"],model_store["LightGBM (calibrated)"]["acc"],
      model_store["Random Forest (calibrated)"]["acc"],ens_acc,pois_acc]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Model Comparison",fontsize=13,fontweight="bold")
colors_=["#2ecc71" if ll==min(lls) else "#3498db" for ll in lls]
bars1=ax1.bar(names,lls,color=colors_,edgecolor="white")
ax1.set_title("Log-Loss (lower=better)"); ax1.set_ylabel("Log-Loss")
for bar,val in zip(bars1,lls): ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f"{val:.4f}",ha="center",fontsize=8)
ax1.tick_params(axis="x",rotation=35); ax1.spines[["top","right"]].set_visible(False)
bars2=ax2.bar(names,[a*100 for a in accs],color=colors_,edgecolor="white")
ax2.set_title("Accuracy % (higher=better)"); ax2.set_ylabel("Accuracy (%)")
for bar,val in zip(bars2,accs): ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f"{val*100:.1f}%",ha="center",fontsize=8)
ax2.tick_params(axis="x",rotation=35); ax2.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "model_comparison.png"),dpi=150,bbox_inches="tight")
plt.show()
print("All charts saved to outputs/")







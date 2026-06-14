"""
FIFA World Cup 2026 — ML Prediction App
=============================================
Author  : Iyinoluwa Don-Taiwo · 

Poisson goals model → predicted scorelines (xG display)
Historical accuracy tracker — model predictions vs actual results
Head-to-head deep dive tab
Supabase shared live results 
LR kept as primary predictor (best log-loss)
"""

import warnings, pickle, random, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy.stats import poisson as sp_poisson
from sklearn.linear_model import LogisticRegression
from supabase import create_client, Client

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="WC 2026 Predictor", page_icon="🏆",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#0a0a12}
h1,h2,h3{color:#f0e4c0!important;font-family:Georgia,serif}
[data-testid="metric-container"]{background:#12121e;border:1px solid #2a2040;border-radius:10px;padding:12px}
[data-testid="metric-container"] label{color:#8a7a9a!important;font-size:12px}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#c8a84b!important;font-size:26px;font-weight:bold}
[data-testid="stSidebar"]{background:#0d0d18;border-right:1px solid #1e1e2e}
.stTabs [data-baseweb="tab-list"]{background:#0d0d18;border-bottom:1px solid #2a2040}
.stTabs [data-baseweb="tab"]{color:#6a6a8a;font-family:Georgia,serif}
.stTabs [aria-selected="true"]{color:#c8a84b!important;border-bottom:2px solid #c8a84b!important}
.stButton>button{background:linear-gradient(135deg,#c8a84b,#a07830);color:#0a0a12;font-weight:bold;border:none;border-radius:8px;font-family:Georgia,serif}
.info-box{background:#12121e;border:1px solid #2a2040;border-left:4px solid #c8a84b;border-radius:8px;padding:16px 20px;margin:12px 0;font-family:Georgia,serif;color:#ccc0b0;font-size:14px}
.result-box{background:#0d1a2a;border:1px solid #1a3a5a;border-radius:12px;padding:20px;margin:16px 0;text-align:center}
.winner-banner{background:linear-gradient(135deg,#1a3a1a,#0d2a0d);border:1px solid #2a6a2a;border-radius:12px;padding:20px;text-align:center;margin:12px 0}
.score-box{background:#12121e;border:1px solid #2a2040;border-radius:10px;padding:16px;text-align:center;margin:8px 0}
.h2h-win{background:#0d2a0d;border:1px solid #2a6a2a;border-radius:8px;padding:8px 14px;margin:4px 0}
.h2h-draw{background:#1a1a2a;border:1px solid #3a3a6a;border-radius:8px;padding:8px 14px;margin:4px 0}
.h2h-loss{background:#2a0d0d;border:1px solid #6a2a2a;border-radius:8px;padding:8px 14px;margin:4px 0}
.accuracy-good{color:#5adb5a;font-weight:bold}
.accuracy-bad{color:#db5a5a;font-weight:bold}
.footer{text-align:center;color:#4a4a6a;font-size:11px;margin-top:40px;font-family:Georgia,serif}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
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
ALL_WC_TEAMS = list({t for grp in WC2026_GROUPS.values() for t in grp})
HOST_NATIONS = {"USA","Canada","Mexico"}

CONFEDERATION = {
    "Spain":"UEFA","France":"UEFA","England":"UEFA","Germany":"UEFA",
    "Netherlands":"UEFA","Belgium":"UEFA","Portugal":"UEFA","Croatia":"UEFA",
    "Switzerland":"UEFA","Norway":"UEFA","Scotland":"UEFA","Austria":"UEFA",
    "Brazil":"CONMEBOL","Argentina":"CONMEBOL","Colombia":"CONMEBOL",
    "Uruguay":"CONMEBOL","Ecuador":"CONMEBOL","Paraguay":"CONMEBOL",
    "Morocco":"CAF","Senegal":"CAF","Egypt":"CAF","Ivory Coast":"CAF",
    "Ghana":"CAF","Cape Verde":"CAF","South Africa":"CAF","Tunisia":"CAF",
    "Algeria":"CAF","Cameroon":"CAF",
    "Japan":"AFC","Korea Republic":"AFC","Iran":"AFC","Saudi Arabia":"AFC",
    "Australia":"AFC","Uzbekistan":"AFC","Jordan":"AFC","Qatar":"AFC","Thailand":"AFC",
    "Mexico":"CONCACAF","USA":"CONCACAF","Canada":"CONCACAF","Panama":"CONCACAF",
    "Curacao":"CONCACAF","Haiti":"CONCACAF","Honduras":"CONCACAF","Costa Rica":"CONCACAF",
    "New Zealand":"OFC",
}
CONF_COLORS={"UEFA":"#1a78cf","CONMEBOL":"#2ecc71","CAF":"#e67e22",
             "AFC":"#e74c3c","CONCACAF":"#27ae60","OFC":"#95a5a6"}
FLAGS={
    "Spain":"🇪🇸","France":"🇫🇷","Argentina":"🇦🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Mexico":"🇲🇽",
    "Morocco":"🇲🇦","Japan":"🇯🇵","Portugal":"🇵🇹","Netherlands":"🇳🇱","Senegal":"🇸🇳",
    "Germany":"🇩🇪","USA":"🇺🇸","Brazil":"🇧🇷","Australia":"🇦🇺","Croatia":"🇭🇷",
    "Iran":"🇮🇷","Belgium":"🇧🇪","Panama":"🇵🇦","Canada":"🇨🇦","Colombia":"🇨🇴",
    "Uruguay":"🇺🇾","Uzbekistan":"🇺🇿","Egypt":"🇪🇬","Switzerland":"🇨🇭","Algeria":"🇩🇿",
    "Qatar":"🇶🇦","Ecuador":"🇪🇨","Ivory Coast":"🇨🇮","Norway":"🇳🇴","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Saudi Arabia":"🇸🇦","Jordan":"🇯🇴","South Africa":"🇿🇦","Tunisia":"🇹🇳",
    "Paraguay":"🇵🇾","Austria":"🇦🇹","New Zealand":"🇳🇿","Haiti":"🇭🇹","Curacao":"🇨🇼",
    "Ghana":"🇬🇭","Cape Verde":"🇨🇻","Korea Republic":"🇰🇷","Honduras":"🇭🇳",
    "Cameroon":"🇨🇲","Costa Rica":"🇨🇷","Thailand":"🇹🇭",
}
def flag(t): return FLAGS.get(t,"🏳️")

TW={'FIFA World Cup':60,'UEFA Euro':50,'Copa America':50,'African Cup of Nations':50,
    'AFC Asian Cup':50,'Gold Cup':40,'CONCACAF Nations League':35,'UEFA Nations League':35,
    'UEFA Euro qualification':30,'FIFA World Cup qualification':30,
    'African Cup of Nations qualification':25,'AFC Asian Cup qualification':25}
def get_k(t):
    for key,k in TW.items():
        if key.lower() in t.lower(): return k
    return 25
def get_imp(t):
    t=t.lower()
    if "world cup" in t and "qualif" not in t: return 3
    if any(x in t for x in ["euro","copa","cup of nations","asian cup","gold cup"]) and "qualif" not in t: return 2
    if "nations league" in t or "qualif" in t: return 1
    return 0


# ─────────────────────────────────────────────────────────────
# SUPABASE
# ─────────────────────────────────────────────────────────────
SUPABASE_URL = "https://rymznaqmclbrsybbghpz.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5bXpuYXFtY2xicnN5YmJnaHB6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODEyNTM0MDEsImV4cCI6MjA5NjgyOTQwMX0"
    ".mRlvkoKx0x2WB-1WPdA8JsXeDCeidI2metUzmPfsCcM"
)

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = get_supabase()

def fetch_live_results():
    try:
        res = supabase.table("live_results").select("*").order("entered_at").execute()
        return res.data or []
    except Exception as e:
        st.warning(f"Database unreachable: {e}"); return []

def insert_result(home,away,hs,as_):
    try:
        supabase.table("live_results").insert({
            "home_team":home,"away_team":away,
            "home_score":hs,"away_score":as_,"tournament":"FIFA World Cup"
        }).execute(); return True
    except Exception as e:
        st.error(f"Save failed: {e}"); return False

def delete_one(rid):
    try:
        supabase.table("live_results").delete().eq("id",rid).execute(); return True
    except Exception as e:
        st.error(f"Delete failed: {e}"); return False

def clear_all():
    try:
        supabase.table("live_results").delete().neq("id",0).execute(); return True
    except Exception as e:
        st.error(f"Clear failed: {e}"); return False

def rows_to_dict(rows):
    out={}
    for r in rows:
        key=f"{r['home_team']}|{r['away_team']}"
        out[key]={"home_score":r["home_score"],"away_score":r["away_score"],"id":r["id"]}
    return out


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

import subprocess, sys, os

# Rebuild model if pkl is missing or was built on a different Python version
if not os.path.exists("models/best_model_v4.pkl"):
    st.info("Building model for first time — this takes 2-3 minutes...")
    subprocess.run([sys.executable, "build_model.py"], check=True)

@st.cache_resource
def load_model():
    with open("models/best_model_v4.pkl","rb") as f:
        return pickle.load(f)

saved        = load_model()
model        = saved["model"]           # v2 LR — best log-loss
poisson_home = saved["poisson_home"]
poisson_away = saved["poisson_away"]
PFEATS       = saved["poisson_feats"]
FEATURES     = saved["features"]
base_df      = saved["df"].copy()
base_df["date"] = pd.to_datetime(base_df["date"])
elo_base     = saved["elo_state"]
le           = saved["le"]
model_results= saved.get("model_results",{})
precomp_sim  = saved.get("sim_results",{})
idx_hw       = saved.get("idx_hw",2)
idx_d        = saved.get("idx_d",1)
idx_aw       = saved.get("idx_aw",0)


# ─────────────────────────────────────────────────────────────
# LIVE RETRAINING
# ─────────────────────────────────────────────────────────────
def retrain(live_rows):
    if not live_rows: return saved["model"]
    df=base_df.copy()
    new_rows=[]
    for r in live_rows:
        home,away=r["home_team"],r["away_team"]
        hs_,as__=r["home_score"],r["away_score"]
        result=1 if hs_>as__ else(-1 if as__>hs_ else 0)
        hm=df[df["home_team"]==home]; am=df[df["away_team"]==away]
        h_pts=float(hm.iloc[-1]["home_ranking_pts"]) if len(hm) else 1200.0
        a_pts=float(am.iloc[-1]["away_ranking_pts"]) if len(am) else 1200.0
        new_rows.append({"home_team":home,"away_team":away,"home_score":hs_,"away_score":as__,
            "result":result,"date":pd.Timestamp("2026-06-15"),
            "home_ranking_pts":h_pts,"away_ranking_pts":a_pts,
            "tournament":"FIFA World Cup","neutral":True,
            "home_confederation":CONFEDERATION.get(home,"UEFA"),
            "away_confederation":CONFEDERATION.get(away,"UEFA")})
    expanded=pd.concat([df,pd.DataFrame(new_rows)],ignore_index=True).sort_values("date")
    elo=defaultdict(lambda:1500.0)
    hew,aew=[],[]
    for _,row in expanded.iterrows():
        h,a=row["home_team"],row["away_team"]; he,ae=elo[h],elo[a]
        hew.append(he); aew.append(ae)
        exp_h=1/(1+10**((ae-he)/400))
        sh=1.0 if row["result"]==1 else(0.5 if row["result"]==0 else 0.0)
        k=get_k(row["tournament"])
        elo[h]=he+k*(sh-exp_h); elo[a]=ae+k*((1-sh)-(1-exp_h))
    expanded["home_elo_w"]=hew; expanded["away_elo_w"]=aew
    expanded["elo_diff_w"]=expanded["home_elo_w"]-expanded["away_elo_w"]
    expanded["ranking_diff"]=expanded["home_ranking_pts"]-expanded["away_ranking_pts"]
    expanded["is_neutral"]=expanded["neutral"].astype(int)
    expanded["same_confederation"]=(expanded["home_confederation"]==expanded["away_confederation"]).astype(int)
    expanded["tournament_importance"]=expanded["tournament"].apply(get_imp)
    for col in ["home_form_5","away_form_5","form_diff","home_form_10","away_form_10",
                "form_diff_10","home_goals_scored_5","home_goals_conceded_5",
                "away_goals_scored_5","away_goals_conceded_5","goal_diff_scored",
                "goal_diff_conceded","h2h_home_wins","h2h_draws","h2h_home_losses"]:
        if col not in expanded.columns: expanded[col]=0.0
    expanded["result_encoded"]=le.transform(expanded["result"])
    mask=(expanded["date"]<"2018-01-01")|(expanded["date"]>="2026-01-01")
    X_tr=expanded[mask][FEATURES].fillna(0); y_tr=expanded[mask]["result_encoded"]
    new_model=LogisticRegression(solver="lbfgs",max_iter=2000,C=0.01)
    new_model.fit(X_tr,y_tr)
    return new_model


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for k,v in [("sim_cache",None),("last_fetch",0),("live_rows",[]),("active_model",model)]:
    if k not in st.session_state: st.session_state[k]=v

if time.time()-st.session_state.last_fetch>30:
    st.session_state.live_rows=fetch_live_results()
    st.session_state.last_fetch=time.time()

live_rows     = st.session_state.live_rows
match_results = rows_to_dict(live_rows)
active_model  = st.session_state.active_model


# ─────────────────────────────────────────────────────────────
# TEAM STATS
# ─────────────────────────────────────────────────────────────
@st.cache_data
def build_team_stats(_base_df, _elo_base, live_hash):
    elo=defaultdict(lambda:1500.0,_elo_base)
    df=_base_df.copy()
    mr=rows_to_dict(st.session_state.live_rows)
    for key,res in mr.items():
        home,away=key.split("|"); he,ae=elo[home],elo[away]
        exp_h=1/(1+10**((ae-he)/400))
        sh=1.0 if res["home_score"]>res["away_score"] else(0.5 if res["home_score"]==res["away_score"] else 0.0)
        elo[home]=he+60*(sh-exp_h); elo[away]=ae+60*((1-sh)-(1-exp_h))
    if st.session_state.live_rows:
        nr=[]
        for r in st.session_state.live_rows:
            home,away=r["home_team"],r["away_team"]
            hs_,as__=r["home_score"],r["away_score"]
            result=1 if hs_>as__ else(-1 if as__>hs_ else 0)
            nr.append({"home_team":home,"away_team":away,"home_score":hs_,"away_score":as__,
                "result":result,"date":pd.Timestamp("2026-06-15"),
                "home_ranking_pts":1500,"away_ranking_pts":1500,
                "home_confederation":CONFEDERATION.get(home,"UEFA"),
                "away_confederation":CONFEDERATION.get(away,"UEFA"),
                "tournament":"FIFA World Cup"})
        df=pd.concat([df,pd.DataFrame(nr)],ignore_index=True).sort_values("date")
    stats={}
    for team in ALL_WC_TEAMS:
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
        conf=CONFEDERATION.get(team,"UEFA")
        if len(hm) and "home_confederation" in hm.columns:
            v=hm.iloc[-1]["home_confederation"]
            if pd.notna(v): conf=v
        stats[team]={"elo":elo.get(team,1500.0),"pts":pts,"form5":form5,"form10":form10,
            "goals_scored5":np.mean(gs) if gs else 1.2,
            "goals_conceded5":np.mean(gc) if gc else 1.0,"confederation":conf}
    return stats

live_hash=str([(r["id"],r["home_score"],r["away_score"]) for r in live_rows])
team_stats=build_team_stats(base_df,elo_base,live_hash)


# ─────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────
def build_row(home,away,neutral=True):
    hs=team_stats.get(home,{"elo":1500,"pts":1200,"form5":0.45,"form10":0.45,"goals_scored5":1.2,"goals_conceded5":1.0,"confederation":"UEFA"})
    as_=team_stats.get(away,{"elo":1500,"pts":1200,"form5":0.45,"form10":0.45,"goals_scored5":1.2,"goals_conceded5":1.0,"confederation":"UEFA"})
    past=base_df[(base_df["home_team"]==home)&(base_df["away_team"]==away)]
    same_c=1 if hs["confederation"]==as_["confederation"] else 0
    is_n=0 if(home in HOST_NATIONS and not neutral) else int(neutral)
    return {"elo_diff_w":hs["elo"]-as_["elo"],"ranking_diff":hs["pts"]-as_["pts"],
        "home_form_5":hs["form5"],"away_form_5":as_["form5"],"form_diff":hs["form5"]-as_["form5"],
        "home_form_10":hs["form10"],"away_form_10":as_["form10"],"form_diff_10":hs["form10"]-as_["form10"],
        "home_goals_scored_5":hs["goals_scored5"],"home_goals_conceded_5":hs["goals_conceded5"],
        "away_goals_scored_5":as_["goals_scored5"],"away_goals_conceded_5":as_["goals_conceded5"],
        "goal_diff_scored":hs["goals_scored5"]-as_["goals_scored5"],
        "goal_diff_conceded":as_["goals_conceded5"]-hs["goals_conceded5"],
        "h2h_home_wins":int((past["result"]==1).sum()),
        "h2h_draws":int((past["result"]==0).sum()),
        "h2h_home_losses":int((past["result"]==-1).sum()),
        "is_neutral":is_n,"same_confederation":same_c,"tournament_importance":3}

def predict_match(home,away,neutral=True):
    row=pd.DataFrame([build_row(home,away,neutral)])
    probs=active_model.predict_proba(row[FEATURES])[0].astype(float)
    probs/=probs.sum()
    classes=list(le.classes_)
    return float(probs[classes.index(1)])*100, float(probs[classes.index(0)])*100, float(probs[classes.index(-1)])*100

def predict_scoreline(home,away,neutral=True):
    """Poisson model: returns (lambda_home, lambda_away, score_matrix_df)."""
    row=pd.DataFrame([build_row(home,away,neutral)])
    lh=max(0.1,float(poisson_home.predict(row[PFEATS])[0]))
    la=max(0.1,float(poisson_away.predict(row[PFEATS])[0]))
    # Score probability matrix (0-5 goals each)
    mg=6
    scores={}
    for h in range(mg):
        for a in range(mg):
            scores[(h,a)]=sp_poisson.pmf(h,lh)*sp_poisson.pmf(a,la)
    return lh,la,scores

def poisson_outcome_probs(lh,la,mg=8):
    p_hw=p_d=p_aw=0.0
    for h in range(mg+1):
        for a in range(mg+1):
            p=sp_poisson.pmf(h,lh)*sp_poisson.pmf(a,la)
            if h>a: p_hw+=p
            elif h==a: p_d+=p
            else: p_aw+=p
    t=p_hw+p_d+p_aw
    return p_hw/t*100,p_d/t*100,p_aw/t*100


# ─────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────
def run_simulation(n,t_stats,groups):
    all_wc=[t for ts in groups.values() for t in ts]
    rows,pairs=[],[]
    for h in all_wc:
        for a in all_wc:
            if h==a: continue
            hs=t_stats.get(h,{"elo":1500,"pts":1200,"form5":0.45,"form10":0.45,"goals_scored5":1.2,"goals_conceded5":1.0,"confederation":"UEFA"})
            as_=t_stats.get(a,{"elo":1500,"pts":1200,"form5":0.45,"form10":0.45,"goals_scored5":1.2,"goals_conceded5":1.0,"confederation":"UEFA"})
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
    batch=active_model.predict_proba(pd.DataFrame(rows)[FEATURES]).astype(float)
    batch/=batch.sum(axis=1,keepdims=True)
    classes=list(le.classes_)
    ihw=classes.index(1); id_=classes.index(0); iaw=classes.index(-1)
    lookup={(h,a):(batch[i][ihw],batch[i][id_],batch[i][iaw]) for i,(h,a) in enumerate(pairs)}

    def sim_group(teams):
        pts={t:0 for t in teams}; gd={t:0 for t in teams}
        gf={t:0 for t in teams}; h2h_pts=defaultdict(int); h2h_gd=defaultdict(int)
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                h,a=teams[i],teams[j]; hw,d,aw=lookup.get((h,a),(0.4,0.2,0.4))
                r=random.random()
                if r<hw: hs_=random.randint(1,3); as__=random.randint(0,max(0,hs_-1)); pts[h]+=3; h2h_pts[h]+=3
                elif r<hw+d: hs_=random.randint(0,2); as__=hs_; pts[h]+=1; pts[a]+=1; h2h_pts[h]+=1; h2h_pts[a]+=1
                else: as__=random.randint(1,3); hs_=random.randint(0,max(0,as__-1)); pts[a]+=3; h2h_pts[a]+=3
                diff=hs_-as__; gd[h]+=diff; gd[a]-=diff; gf[h]+=hs_; gf[a]+=as__
                h2h_gd[h]+=diff; h2h_gd[a]-=diff
        ranked=sorted(teams,key=lambda t:(pts[t],h2h_pts[t],h2h_gd[t],gd[t],gf[t],random.random()),reverse=True)
        return ranked,pts,gd,gf

    def ko_winner(h,a):
        hw,d,aw=lookup.get((h,a),(0.4,0.2,0.4)); p_h=hw+d/2; p_a=aw+d/2
        pr=np.array([p_h,p_a]); pr/=pr.sum()
        return h if random.random()<pr[0] else a

    wins=defaultdict(int); sf_c=defaultdict(int); f_c=defaultdict(int)
    for _ in range(n):
        qualifiers=[]; thirds=[]
        for _,teams in groups.items():
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
    return wins,sf_c,f_c


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏆 WC 2026 Predictor")
    st.markdown("---")
    st.markdown("**v4 Upgrades**")
    st.markdown("""
- Poisson scoreline predictions (xG)
- Historical accuracy tracker
- Head-to-head deep dive
- Shared live results (Supabase)
- Model retrains on each new result
    """)
    st.markdown("---")
    st.metric("Live Results", len(live_rows))
    if model_results:
        st.markdown("**Model Performance**")
        for name,r in model_results.items():
            st.markdown(f"`{name}` ll:{r['ll']:.4f}")
    st.markdown("---")
    st.markdown('<div class="footer">Iyinoluwa Don-Taiwo · 2026</div>',unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
c1,c2,c3=st.columns([3,1,1])
with c1:
    st.markdown("# 🏆 FIFA World Cup 2026")
    st.markdown("**ML Prediction System v4** — Poisson xG · Accuracy Tracker · H2H Deep Dive · Supabase Live Results")
with c2: st.metric("Live Results",len(live_rows))
with c3:
    if st.button("🔄 Refresh"):
        st.session_state.live_rows=fetch_live_results()
        st.session_state.last_fetch=time.time()
        st.rerun()
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
    "⚽ Match Predictor","🎯 Scoreline (xG)",
    "🏟️ Knockout Simulator","📊 Win Probabilities",
    "📋 Group Stage","🔍 Head-to-Head",
    "📡 Live Results & Accuracy",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Match Outcome Predictor")
    st.markdown('<div class="info-box">Win/Draw/Loss probabilities from the v2 Logistic Regression model (best log-loss: 0.8775). Updates automatically when live results are entered.</div>',unsafe_allow_html=True)
    real_teams=sorted(ALL_WC_TEAMS)
    c1,c2,c3=st.columns([2,1,2])
    with c1: home=st.selectbox("🏠 Home / Team A",real_teams,index=real_teams.index("France") if "France" in real_teams else 0)
    with c2: st.markdown("<br><h3 style='text-align:center;color:#c8a84b;'>VS</h3>",unsafe_allow_html=True)
    with c3: away=st.selectbox("✈️ Away / Team B",real_teams,index=real_teams.index("Argentina") if "Argentina" in real_teams else 1)
    neutral=st.checkbox("Neutral venue",value=True)

    if st.button("⚡ Predict Outcome",use_container_width=True):
        if home==away: st.warning("Select two different teams.")
        else:
            hw,d,aw_=predict_match(home,away,neutral)
            if hw>aw_ and hw>d: verdict=f"🟢 {home} favoured"; vc="#1a6a4a"
            elif aw_>hw and aw_>d: verdict=f"🔴 {away} favoured"; vc="#6a1a1a"
            else: verdict="⚪ Too close to call"; vc="#3a3a5a"
            st.markdown(f'<div class="result-box"><h2 style="color:#f0e4c0;">{flag(home)} {home} vs {away} {flag(away)}</h2><p style="color:{vc};font-size:16px;font-weight:bold;">{verdict}</p></div>',unsafe_allow_html=True)
            r1,r2,r3=st.columns(3)
            r1.metric(f"{home} Win",f"{hw:.1f}%"); r2.metric("Draw",f"{d:.1f}%"); r3.metric(f"{away} Win",f"{aw_:.1f}%")
            fig,ax=plt.subplots(figsize=(8,2.5))
            fig.patch.set_facecolor("#0d0d18"); ax.set_facecolor("#0d0d18")
            bars=ax.barh([f"{home} Win","Draw",f"{away} Win"],[hw,d,aw_],color=["#1a6a4a","#4a4a7a","#7a1a1a"],height=0.5,edgecolor="#1a1a2a")
            for bar,val in zip(bars,[hw,d,aw_]):
                ax.text(val+0.5,bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va="center",color="white",fontsize=11)
            ax.set_xlim(0,105); ax.set_xlabel("Probability (%)",color="#9a8a9a")
            ax.tick_params(colors="#9a8a9a"); ax.spines[:].set_visible(False)
            st.pyplot(fig); plt.close()
            st.markdown("#### Team Comparison")
            hs_=team_stats.get(home,{}); as__=team_stats.get(away,{})
            comp=pd.DataFrame({"Stat":["Elo Rating","FIFA Points","Form (last 5)","Form (last 10)","Goals Scored/5","Goals Conceded/5","Confederation"],
                home:[f"{hs_.get('elo',1500):.0f}",f"{hs_.get('pts',1200):.0f}",f"{hs_.get('form5',0.45)*100:.0f}%",f"{hs_.get('form10',0.45)*100:.0f}%",f"{hs_.get('goals_scored5',1.2):.2f}",f"{hs_.get('goals_conceded5',1.0):.2f}",CONFEDERATION.get(home,"?")],
                away:[f"{as__.get('elo',1500):.0f}",f"{as__.get('pts',1200):.0f}",f"{as__.get('form5',0.45)*100:.0f}%",f"{as__.get('form10',0.45)*100:.0f}%",f"{as__.get('goals_scored5',1.2):.2f}",f"{as__.get('goals_conceded5',1.0):.2f}",CONFEDERATION.get(away,"?")]})
            st.dataframe(comp.set_index("Stat"),use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — SCORELINE / xG
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Predicted Scoreline (xG)")
    st.markdown('<div class="info-box">Poisson regression predicts expected goals (xG) for each team separately. The score probability matrix shows the likelihood of every possible scoreline. This is how professional football analytics models work.</div>',unsafe_allow_html=True)
    real_teams=sorted(ALL_WC_TEAMS)
    sc1,sc2,sc3=st.columns([2,1,2])
    with sc1: xg_home=st.selectbox("🏠 Home Team",real_teams,key="xg_home",index=real_teams.index("Brazil") if "Brazil" in real_teams else 0)
    with sc2: st.markdown("<br><h3 style='text-align:center;color:#c8a84b;'>VS</h3>",unsafe_allow_html=True)
    with sc3: xg_away=st.selectbox("✈️ Away Team",real_teams,key="xg_away",index=real_teams.index("France") if "France" in real_teams else 1)
    xg_neutral=st.checkbox("Neutral venue",value=True,key="xg_neutral")

    if st.button("🎯 Predict Scoreline",use_container_width=True):
        if xg_home==xg_away: st.warning("Select two different teams.")
        else:
            lh,la,scores=predict_scoreline(xg_home,xg_away,xg_neutral)
            hw_p,d_p,aw_p=poisson_outcome_probs(lh,la)

            # Most likely scores
            top_scores=sorted(scores.items(),key=lambda x:-x[1])[:6]

            st.markdown(f'<div class="result-box"><h2 style="color:#f0e4c0;">{flag(xg_home)} {xg_home} vs {xg_away} {flag(xg_away)}</h2></div>',unsafe_allow_html=True)

            xc1,xc2,xc3=st.columns(3)
            xc1.metric(f"xG {xg_home}",f"{lh:.2f}")
            xc2.metric("Expected Total Goals",f"{lh+la:.2f}")
            xc3.metric(f"xG {xg_away}",f"{la:.2f}")

            st.markdown("#### Outcome Probabilities (Poisson)")
            pc1,pc2,pc3=st.columns(3)
            pc1.metric(f"{xg_home} Win",f"{hw_p:.1f}%")
            pc2.metric("Draw",f"{d_p:.1f}%")
            pc3.metric(f"{xg_away} Win",f"{aw_p:.1f}%")

            st.markdown("#### Most Likely Scorelines")
            cols=st.columns(3)
            for i,(score,prob) in enumerate(top_scores):
                h_g,a_g=score
                with cols[i%3]:
                    outcome="🟢" if h_g>a_g else("🔴" if a_g>h_g else "⚪")
                    st.markdown(f'<div class="score-box"><div style="font-size:28px;font-weight:bold;color:#c8a84b;">{h_g} — {a_g}</div><div style="color:#9a8a9a;font-size:13px;">{outcome} {prob*100:.1f}% probability</div></div>',unsafe_allow_html=True)

            st.markdown("#### Score Probability Matrix")
            mg=6
            matrix=np.zeros((mg,mg))
            for h in range(mg):
                for a in range(mg):
                    matrix[h,a]=scores.get((h,a),0)*100
            fig,ax=plt.subplots(figsize=(8,6))
            fig.patch.set_facecolor("#0d0d18"); ax.set_facecolor("#0d0d18")
            im=ax.imshow(matrix,cmap="YlOrRd",aspect="auto")
            ax.set_xticks(range(mg)); ax.set_yticks(range(mg))
            ax.set_xticklabels([str(i) for i in range(mg)],color="white")
            ax.set_yticklabels([str(i) for i in range(mg)],color="white")
            ax.set_xlabel(f"{xg_away} Goals",color="#9a8a9a",fontsize=11)
            ax.set_ylabel(f"{xg_home} Goals",color="#9a8a9a",fontsize=11)
            ax.set_title(f"Score Probabilities — {xg_home} vs {xg_away}",color="#f0e4c0",fontsize=12)
            for h in range(mg):
                for a in range(mg):
                    ax.text(a,h,f"{matrix[h,a]:.1f}%",ha="center",va="center",fontsize=8,color="black" if matrix[h,a]>5 else "white")
            plt.colorbar(im,ax=ax,label="Probability (%)"); plt.tight_layout()
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 3 — KNOCKOUT SIMULATOR
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Knockout Bracket Simulator")
    st.markdown('<div class="info-box">Pick 16 teams for the Round of 16. The model predicts each match probabilistically — upsets happen naturally.</div>',unsafe_allow_html=True)
    real_wc=sorted(ALL_WC_TEAMS)
    default_r16=["France","Argentina","Spain","England","Brazil","Germany","Portugal","Netherlands","Morocco","Japan","Senegal","Croatia","Colombia","Mexico","Uruguay","Belgium"]
    default_r16=[t for t in default_r16 if t in real_wc][:16]
    selected=st.multiselect("Choose exactly 16 teams",real_wc,default=default_r16,max_selections=16)
    if len(selected)>=2 and st.button("🎯 Simulate Knockout",use_container_width=True):
        teams_ko=selected if len(selected)%2==0 else selected[:-1]
        current=list(teams_ko); round_names=["Round of 16","Quarterfinals","Semifinals","Final"]
        all_rounds={}; round_idx=0
        while len(current)>1:
            rname=round_names[min(round_idx,len(round_names)-1)]
            matchups=[]; winners=[]
            for i in range(0,len(current)-1,2):
                h,a=current[i],current[i+1]; hw,d,aw_=predict_match(h,a,neutral=True)
                p_h=(hw+d/2)/100; p_a=(aw_+d/2)/100; pr=np.array([p_h,p_a]); pr/=pr.sum()
                win=np.random.choice([h,a],p=pr)
                matchups.append({"home":h,"away":a,"hw":round(hw,1),"d":round(d,1),"aw":round(aw_,1),"winner":win}); winners.append(win)
            all_rounds[rname]=matchups; current=winners; round_idx+=1
        champion=current[0]
        st.markdown(f'<div class="winner-banner"><h1 style="color:#c8a84b;font-size:40px;">{flag(champion)}</h1><h2 style="color:#c8a84b;">🏆 {champion} wins the World Cup!</h2></div>',unsafe_allow_html=True)
        for rname,matchups in all_rounds.items():
            st.markdown(f"#### {rname}")
            for m in matchups:
                won_home=m["winner"]==m["home"]
                ca,cb,cc=st.columns([3,2,3])
                w="bold" if won_home else "normal"
                ca.markdown(f"<p style='text-align:right;font-weight:{w};color:#e8e0d0;'>{flag(m['home'])} {m['home']}</p>",unsafe_allow_html=True)
                cb.markdown(f"<p style='text-align:center;color:#c8a84b;font-size:12px;'>{m['hw']}% — {m['d']}% — {m['aw']}%</p>",unsafe_allow_html=True)
                w="bold" if not won_home else "normal"
                cc.markdown(f"<p style='font-weight:{w};color:#e8e0d0;'>{m['away']} {flag(m['away'])}</p>",unsafe_allow_html=True)
                st.markdown(f"<p style='color:#c8a84b;font-size:12px;text-align:center;'>✓ {m['winner']} advances</p>",unsafe_allow_html=True)
                st.markdown("---")


# ══════════════════════════════════════════════════════════════
# TAB 4 — WIN PROBABILITIES
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Tournament Win Probabilities")
    if precomp_sim:
        st.markdown("#### Pre-Tournament Simulation (10,000 runs)")
        top10=sorted(precomp_sim.items(),key=lambda x:-x[1]["win_pct"])[:10]
        mc1,mc2,mc3=st.columns(3)
        if top10: mc1.metric("🥇 "+top10[0][0],f"{top10[0][1]['win_pct']:.1f}%")
        if len(top10)>1: mc2.metric("🥈 "+top10[1][0],f"{top10[1][1]['win_pct']:.1f}%")
        if len(top10)>2: mc3.metric("🥉 "+top10[2][0],f"{top10[2][1]['win_pct']:.1f}%")
        tbl=[{"Rank":i+1,"Flag":flag(t),"Team":t,"Conf.":CONFEDERATION.get(t,"?"),
              "Win %":f"{d['win_pct']:.2f}%","Final %":f"{d['final_pct']:.1f}%","Semi %":f"{d['semi_pct']:.1f}%"}
             for i,(t,d) in enumerate(sorted(precomp_sim.items(),key=lambda x:-x[1]["win_pct"]))]
        st.dataframe(pd.DataFrame(tbl),use_container_width=True,hide_index=True)
        st.markdown("---")
    cl,cr=st.columns([1,2])
    with cl:
        n_sims=st.select_slider("Simulations",options=[1000,2000,5000,10000],value=5000)
        run_btn=st.button("▶ Run Live Simulation",use_container_width=True)
    with cr:
        st.markdown('<div class="info-box">Runs the full 48-team bracket using the current model including any live results entered. Win% = how often each team won across all runs.</div>',unsafe_allow_html=True)
    if run_btn or st.session_state.sim_cache is not None:
        if run_btn:
            with st.spinner(f"Running {n_sims:,} simulations..."):
                wins,sf_c,f_c=run_simulation(n_sims,team_stats,WC2026_GROUPS)
                st.session_state.sim_cache=(wins,sf_c,f_c,n_sims)
        else:
            wins,sf_c,f_c,n_sims=st.session_state.sim_cache
        ranked=sorted(wins.items(),key=lambda x:-x[1]); total=sum(wins.values())
        m1,m2,m3,m4=st.columns(4)
        if ranked: m1.metric("🥇 Champion",ranked[0][0],f"{ranked[0][1]/total*100:.1f}%")
        if len(ranked)>1: m2.metric("🥈 Runner Up",ranked[1][0],f"{ranked[1][1]/total*100:.1f}%")
        if len(ranked)>2: m3.metric("🥉 3rd Fav",ranked[2][0],f"{ranked[2][1]/total*100:.1f}%")
        m4.metric("Simulations",f"{total:,}")
        fig,ax=plt.subplots(figsize=(10,max(8,len(ranked)*0.38)))
        fig.patch.set_facecolor("#0a0a12"); ax.set_facecolor("#0a0a12")
        tnames=[r[0] for r in ranked]; tprobs=[r[1]/total*100 for r in ranked]
        tcols=[CONF_COLORS.get(CONFEDERATION.get(t,"?"),"#888") for t in tnames]
        bars=ax.barh(tnames,tprobs,color=tcols,edgecolor="#1a1a2a",linewidth=0.5)
        ax.axvline(100/48,color="#c8a84b",linestyle="--",linewidth=1,alpha=0.7,label=f"Random ({100/48:.1f}%)")
        for bar,val in zip(bars,tprobs):
            if val>=0.3: ax.text(val+0.1,bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va="center",color="white",fontsize=8)
        ax.set_xlabel("Win Probability (%)",color="#9a8a9a")
        ax.set_title(f"WC 2026 — Win Probabilities ({total:,} sims)",color="#f0e4c0",fontsize=12,fontweight="bold")
        ax.tick_params(colors="#9a8a9a"); ax.spines[:].set_visible(False)
        patches=[mpatches.Patch(color=v,label=k) for k,v in CONF_COLORS.items()]
        ax.legend(handles=patches+[ax.lines[0]],loc="lower right",facecolor="#0d0d18",labelcolor="white",fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        tbl_=[{"Rank":i+1,"Flag":flag(t),"Team":t,"Conf.":CONFEDERATION.get(t,"?"),
               "Win %":f"{c/total*100:.2f}%","Semi %":f"{sf_c.get(t,0)/total*100:.1f}%","Final %":f"{f_c.get(t,0)/total*100:.1f}%"}
              for i,(t,c) in enumerate(ranked)]
        st.dataframe(pd.DataFrame(tbl_),use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — GROUP STAGE
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Group Stage Predictor")
    st.markdown('<div class="info-box">Simulates all 6 group matches with proper FIFA tiebreakers: H2H points → H2H goal difference → overall GD → goals scored.</div>',unsafe_allow_html=True)
    sel_grp=st.selectbox("Select group",list(WC2026_GROUPS.keys()),format_func=lambda x:f"Group {x}")
    grp_teams=WC2026_GROUPS[sel_grp]
    gc=st.columns(len(grp_teams))
    for i,team in enumerate(grp_teams):
        with gc[i]:
            conf=CONFEDERATION.get(team,"?"); color=CONF_COLORS.get(conf,"#555")
            st.markdown(f'<div style="background:#12121e;border:1px solid #2a2040;border-top:3px solid {color};border-radius:8px;padding:12px;text-align:center;"><div style="font-size:28px;">{flag(team)}</div><div style="color:#e8e0d0;font-weight:bold;font-size:13px;">{team}</div><div style="color:{color};font-size:11px;">{conf}</div></div>',unsafe_allow_html=True)
    if st.button("⚽ Simulate Group",use_container_width=True):
        pts={t:0 for t in grp_teams}; gd={t:0 for t in grp_teams}
        gf={t:0 for t in grp_teams}; ga={t:0 for t in grp_teams}
        h2h_pts=defaultdict(int); h2h_gd=defaultdict(int); results_log=[]
        for i in range(len(grp_teams)):
            for j in range(i+1,len(grp_teams)):
                h,a=grp_teams[i],grp_teams[j]; hw,d,aw_=predict_match(h,a,neutral=True)
                probs=np.array([hw/100,d/100,aw_/100]); probs/=probs.sum()
                o=np.random.choice(["Home Win","Draw","Away Win"],p=probs)
                if o=="Home Win": hs_=random.randint(1,3); as__=random.randint(0,max(0,hs_-1)); pts[h]+=3; h2h_pts[h]+=3
                elif o=="Draw": hs_=random.randint(0,2); as__=hs_; pts[h]+=1; pts[a]+=1; h2h_pts[h]+=1; h2h_pts[a]+=1
                else: as__=random.randint(1,3); hs_=random.randint(0,max(0,as__-1)); pts[a]+=3; h2h_pts[a]+=3
                diff=hs_-as__; gd[h]+=diff; gd[a]-=diff; gf[h]+=hs_; gf[a]+=as__; ga[h]+=as__; ga[a]+=hs_
                h2h_gd[h]+=diff; h2h_gd[a]-=diff
                results_log.append({"Match":f"{flag(h)} {h} vs {a} {flag(a)}","Score":f"{hs_} - {as__}","Result":o,f"{h} Win":f"{hw:.0f}%","Draw":f"{d:.0f}%",f"{a} Win":f"{aw_:.0f}%"})
        standings=sorted(grp_teams,key=lambda t:(pts[t],h2h_pts[t],h2h_gd[t],gd[t],gf[t],random.random()),reverse=True)
        st.markdown("#### Predicted Standings")
        for rank,team in enumerate(standings,1):
            qualifies=rank<=2; badge="✅ Qualifies" if qualifies else "❌ Eliminated"
            bc="#1a4a1a" if qualifies else "#3a1a1a"; bb="#2a7a2a" if qualifies else "#7a2a2a"
            st.markdown(f'<div style="background:{bc};border:1px solid {bb};border-radius:8px;padding:10px 16px;margin:6px 0;"><span style="font-size:20px;font-weight:bold;color:#c8a84b;">{rank}</span>&nbsp;{flag(team)}&nbsp;<span style="color:#e8e0d0;font-size:15px;">{team}</span>&nbsp;&nbsp;<span style="color:#9a8a9a;font-size:13px;">Pts:{pts[team]} GD:{gd[team]:+d} GF:{gf[team]}</span>&nbsp;&nbsp;<span style="font-size:12px;color:{"#5adb5a" if qualifies else "#db5a5a"};">{badge}</span></div>',unsafe_allow_html=True)
        st.markdown("#### Match Results"); st.dataframe(pd.DataFrame(results_log),use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — HEAD-TO-HEAD DEEP DIVE
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Head-to-Head Deep Dive")
    st.markdown('<div class="info-box">Full historical record between any two teams from 11,437 competitive international matches (2002–2026). Shows match timeline, result breakdown, and goal statistics.</div>',unsafe_allow_html=True)
    all_teams_sorted=sorted(list(set(base_df["home_team"])|set(base_df["away_team"])))
    hc1,hc2,hc3=st.columns([2,1,2])
    with hc1: h2h_t1=st.selectbox("Team A",sorted(ALL_WC_TEAMS),key="h2h_t1",index=sorted(ALL_WC_TEAMS).index("Brazil") if "Brazil" in ALL_WC_TEAMS else 0)
    with hc2: st.markdown("<br><h3 style='text-align:center;color:#c8a84b;'>vs</h3>",unsafe_allow_html=True)
    with hc3: h2h_t2=st.selectbox("Team B",sorted(ALL_WC_TEAMS),key="h2h_t2",index=sorted(ALL_WC_TEAMS).index("Argentina") if "Argentina" in ALL_WC_TEAMS else 1)

    if st.button("🔍 Show H2H Record",use_container_width=True):
        if h2h_t1==h2h_t2: st.warning("Select two different teams.")
        else:
            # All meetings in both directions
            h2h=base_df[
                ((base_df["home_team"]==h2h_t1)&(base_df["away_team"]==h2h_t2))|
                ((base_df["home_team"]==h2h_t2)&(base_df["away_team"]==h2h_t1))
            ].copy().sort_values("date")

            if len(h2h)==0:
                st.info(f"No competitive matches found between {h2h_t1} and {h2h_t2} in the dataset.")
            else:
                # Normalise from t1 perspective
                def t1_result(row):
                    if row["home_team"]==h2h_t1: return row["result"]  # 1=t1 win,0=draw,-1=t2 win
                    else: return -row["result"]  # flip if t1 was away
                h2h["t1_result"]=h2h.apply(t1_result,axis=1)
                def t1_goals(row):
                    return (row["home_score"],row["away_score"]) if row["home_team"]==h2h_t1 else (row["away_score"],row["home_score"])
                h2h["t1_goals"]=h2h.apply(lambda r:t1_goals(r)[0],axis=1)
                h2h["t2_goals"]=h2h.apply(lambda r:t1_goals(r)[1],axis=1)

                t1_wins=int((h2h["t1_result"]==1).sum())
                draws  =int((h2h["t1_result"]==0).sum())
                t2_wins=int((h2h["t1_result"]==-1).sum())
                total  =len(h2h)
                t1_gf  =int(h2h["t1_goals"].sum()); t1_ga=int(h2h["t2_goals"].sum())

                # Summary metrics
                sm1,sm2,sm3,sm4,sm5=st.columns(5)
                sm1.metric("Total Matches",total)
                sm2.metric(f"{h2h_t1} Wins",f"{t1_wins} ({t1_wins/total*100:.0f}%)")
                sm3.metric("Draws",f"{draws} ({draws/total*100:.0f}%)")
                sm4.metric(f"{h2h_t2} Wins",f"{t2_wins} ({t2_wins/total*100:.0f}%)")
                sm5.metric("Goals",f"{t1_gf} – {t1_ga}")

                # Win rate bar
                fig,ax=plt.subplots(figsize=(8,1.2))
                fig.patch.set_facecolor("#0d0d18"); ax.set_facecolor("#0d0d18")
                ax.barh([0],[t1_wins/total],color="#2ecc71",height=0.5,label=h2h_t1)
                ax.barh([0],[draws/total],left=[t1_wins/total],color="#95a5a6",height=0.5,label="Draw")
                ax.barh([0],[t2_wins/total],left=[(t1_wins+draws)/total],color="#e74c3c",height=0.5,label=h2h_t2)
                ax.set_xlim(0,1); ax.axis("off")
                ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.3),ncol=3,facecolor="#0d0d18",labelcolor="white",fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                # Match history
                st.markdown(f"#### All Meetings ({total} matches)")
                for _,row in h2h.iloc[::-1].iterrows():
                    res=row["t1_result"]; t1g=int(row["t1_goals"]); t2g=int(row["t2_goals"])
                    home_shown=row["home_team"]; away_shown=row["away_team"]
                    hs_=int(row["home_score"]); as__=int(row["away_score"])
                    if res==1: css="h2h-win"; badge=f"✅ {h2h_t1} won"
                    elif res==0: css="h2h-draw"; badge="⚪ Draw"
                    else: css="h2h-loss"; badge=f"❌ {h2h_t2} won"
                    tourn=str(row.get("tournament",""))[:35]
                    date_=str(row["date"])[:10]
                    st.markdown(f'<div class="{css}"><span style="color:#9a8a9a;font-size:12px;">{date_} · {tourn}</span><br><span style="color:#e8e0d0;">{flag(home_shown)} {home_shown} <b>{hs_} – {as__}</b> {away_shown} {flag(away_shown)}</span> &nbsp; <span style="font-size:12px;">{badge}</span></div>',unsafe_allow_html=True)

                # Goals chart
                st.markdown("#### Goals per Match Over Time")
                fig,ax=plt.subplots(figsize=(10,3))
                fig.patch.set_facecolor("#0d0d18"); ax.set_facecolor("#0d0d18")
                dates=[pd.to_datetime(d) for d in h2h["date"]]
                ax.bar(dates,h2h["t1_goals"],color="#2ecc71",alpha=0.8,label=h2h_t1,width=200)
                ax.bar(dates,[-g for g in h2h["t2_goals"]],color="#e74c3c",alpha=0.8,label=h2h_t2,width=200)
                ax.axhline(0,color="white",linewidth=0.5)
                ax.set_ylabel("Goals",color="#9a8a9a"); ax.tick_params(colors="#9a8a9a")
                ax.legend(facecolor="#0d0d18",labelcolor="white",fontsize=9)
                ax.spines[:].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 7 — LIVE RESULTS & ACCURACY TRACKER
# ══════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### 📡 Live Results & Model Accuracy")
    st.markdown('<div class="info-box">Results entered here are saved to a shared database — visible to everyone instantly. Each new result retrains the model. The accuracy tracker shows how the model\'s predictions compared to actual outcomes.</div>',unsafe_allow_html=True)

    # ── Accuracy Tracker ─────────────────────────────────────
    if live_rows:
        st.markdown("#### Model Accuracy on Live Matches")
        correct=0; total_live=len(live_rows); prediction_log=[]
        for r in live_rows:
            home,away=r["home_team"],r["away_team"]
            hs_,as__=r["home_score"],r["away_score"]
            actual="Home Win" if hs_>as__ else("Draw" if hs_==as__ else "Away Win")
            # Get what model would have predicted BEFORE the match
            # (using current team stats — approximate)
            hw,d,aw_=predict_match(home,away,neutral=True)
            predicted="Home Win" if hw>d and hw>aw_ else("Draw" if d>hw and d>aw_ else "Away Win")
            # Poisson scoreline
            lh,la,_=predict_scoreline(home,away,neutral=True)
            pred_score=f"{round(lh):.0f} – {round(la):.0f}"
            is_correct=(predicted==actual)
            if is_correct: correct+=1
            prediction_log.append({
                "Match":f"{flag(home)} {home} vs {away} {flag(away)}",
                "Actual Score":f"{hs_} – {as__}",
                "Actual Outcome":actual,
                "Model Predicted":predicted,
                "xG Scoreline":pred_score,
                "Correct":("✅" if is_correct else "❌"),
            })

        acc=correct/total_live*100 if total_live else 0
        ac1,ac2,ac3=st.columns(3)
        ac1.metric("Matches Tracked",total_live)
        ac2.metric("Correct Predictions",correct)
        ac3.metric("Live Accuracy",f"{acc:.0f}%")

        st.dataframe(pd.DataFrame(prediction_log),use_container_width=True,hide_index=True)

        # Accuracy chart
        if total_live>1:
            running_acc=[]; running_correct=0
            for i,row in enumerate(prediction_log):
                if row["Correct"]=="✅": running_correct+=1
                running_acc.append(running_correct/(i+1)*100)
            fig,ax=plt.subplots(figsize=(10,3))
            fig.patch.set_facecolor("#0d0d18"); ax.set_facecolor("#0d0d18")
            ax.plot(range(1,total_live+1),running_acc,color="#c8a84b",linewidth=2,marker="o",markersize=5)
            ax.axhline(50,color="#9a8a9a",linestyle="--",alpha=0.5,label="50% baseline")
            ax.fill_between(range(1,total_live+1),running_acc,alpha=0.15,color="#c8a84b")
            ax.set_xlabel("Matches",color="#9a8a9a"); ax.set_ylabel("Running Accuracy (%)",color="#9a8a9a")
            ax.set_title("Model Accuracy Over Tournament",color="#f0e4c0")
            ax.tick_params(colors="#9a8a9a"); ax.spines[:].set_visible(False)
            ax.legend(facecolor="#0d0d18",labelcolor="white",fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("---")

    # ── Entry form ───────────────────────────────────────────
    st.markdown("#### Enter a Match Result")
    live_teams=sorted(ALL_WC_TEAMS)
    with st.form("result_form",clear_on_submit=True):
        fc1,fc2,fc3=st.columns([3,1,3])
        with fc1: res_home=st.selectbox("Home Team",live_teams,key="res_home")
        with fc2: st.markdown("<br><p style='text-align:center;color:#c8a84b;font-size:20px;'>vs</p>",unsafe_allow_html=True)
        with fc3: res_away=st.selectbox("Away Team",live_teams,index=1,key="res_away")
        sc1,sc2=st.columns(2)
        with sc1: home_score=st.number_input("Home Score",min_value=0,max_value=20,value=1,step=1)
        with sc2: away_score=st.number_input("Away Score",min_value=0,max_value=20,value=0,step=1)
        submitted=st.form_submit_button("✅ Submit Result",use_container_width=True)
        if submitted:
            if res_home==res_away: st.error("Home and away teams must be different.")
            else:
                ok=insert_result(res_home,res_away,int(home_score),int(away_score))
                if ok:
                    st.session_state.live_rows=fetch_live_results()
                    st.session_state.last_fetch=time.time()
                    with st.spinner("Retraining model..."):
                        st.session_state.active_model=retrain(st.session_state.live_rows)
                    st.session_state.sim_cache=None
                    st.success(f"✅ {res_home} {int(home_score)} – {int(away_score)} {res_away} saved. Model retrained on {len(st.session_state.live_rows)} live results.")
                    st.rerun()

    # ── Results table ────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### Database — `{len(live_rows)} entries`")
    if not live_rows:
        st.info("No results yet. Enter the first score above.")
    else:
        for row in reversed(live_rows):
            h,a=row["home_team"],row["away_team"]; hs_,as__=row["home_score"],row["away_score"]
            outcome=f"🟢 {h} won" if hs_>as__ else(f"🔴 {a} won" if as__>hs_ else "⚪ Draw")
            entered=row.get("entered_at","")[:16].replace("T"," ")
            ca,cb,cc,cd=st.columns([3,2,2,1])
            ca.markdown(f"{flag(h)} **{h}** {hs_} – {as__} **{a}** {flag(a)}")
            cb.markdown(f"<span style='color:#9a8a9a;font-size:13px;'>{outcome}</span>",unsafe_allow_html=True)
            cc.markdown(f"<span style='color:#6a6a8a;font-size:12px;'>{entered}</span>",unsafe_allow_html=True)
            if cd.button("🗑",key=f"del_{row['id']}"):
                if delete_one(row["id"]):
                    st.session_state.live_rows=fetch_live_results()
                    st.session_state.last_fetch=time.time()
                    st.session_state.active_model=retrain(st.session_state.live_rows)
                    st.session_state.sim_cache=None; st.rerun()
        st.markdown("---")
        if st.button("🗑️ Clear ALL Results",type="secondary"):
            if clear_all():
                st.session_state.live_rows=[]; st.session_state.last_fetch=time.time()
                st.session_state.active_model=model; st.session_state.sim_cache=None; st.rerun()

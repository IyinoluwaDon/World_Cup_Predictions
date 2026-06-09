"""
FIFA World Cup 2026 — ML Prediction App
==============================================
Author  : Iyinoluwa Don-Taiwo 
Model   : Logistic Regression trained on 11,437 competitive international matches
Features: Weighted Elo · FIFA ranking points · Form-5 & Form-10 · Goals scored/conceded
          Head-to-head · Venue context · Tournament importance
Upgrades: Weighted Elo by match importance · Extended form windows · Goals features
          Hyperparameter-tuned LR · Calibrated ensemble · Improved Monte Carlo simulation
          Proper World Cup tiebreakers · Host-nation home advantage

Run locally : streamlit run wc2026_app.py
Deploy free : streamlit.io/cloud → connect GitHub repo
"""

import warnings, pickle, random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a12; }
    h1, h2, h3 { color: #f0e4c0 !important; font-family: Georgia, serif; }
    [data-testid="metric-container"] {
        background: #12121e; border: 1px solid #2a2040;
        border-radius: 10px; padding: 12px;
    }
    [data-testid="metric-container"] label { color: #8a7a9a !important; font-size: 12px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #c8a84b !important; font-size: 26px; font-weight: bold;
    }
    [data-testid="stSidebar"] { background-color: #0d0d18; border-right: 1px solid #1e1e2e; }
    [data-testid="stSidebar"] .stMarkdown { color: #9a8a9a; }
    .stSelectbox label { color: #9a8a9a !important; font-size: 13px; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0d0d18; border-bottom: 1px solid #2a2040; }
    .stTabs [data-baseweb="tab"] { color: #6a6a8a; font-family: Georgia, serif; letter-spacing: 0.05em; }
    .stTabs [aria-selected="true"] { color: #c8a84b !important; border-bottom: 2px solid #c8a84b !important; }
    .stButton > button {
        background: linear-gradient(135deg, #c8a84b, #a07830);
        color: #0a0a12; font-weight: bold; border: none;
        border-radius: 8px; letter-spacing: 0.08em; font-family: Georgia, serif;
    }
    .stButton > button:hover { opacity: 0.9; }
    .info-box {
        background: #12121e; border: 1px solid #2a2040;
        border-left: 4px solid #c8a84b; border-radius: 8px;
        padding: 16px 20px; margin: 12px 0;
        font-family: Georgia, serif; color: #ccc0b0; font-size: 14px;
    }
    .result-box {
        background: #0d1a2a; border: 1px solid #1a3a5a;
        border-radius: 12px; padding: 20px; margin: 16px 0; text-align: center;
    }
    .winner-banner {
        background: linear-gradient(135deg, #1a3a1a, #0d2a0d);
        border: 1px solid #2a6a2a; border-radius: 12px;
        padding: 20px; text-align: center; margin: 12px 0;
    }
    .upgrade-badge {
        background: #1a1a0a; border: 1px solid #c8a84b;
        border-radius: 6px; padding: 4px 10px;
        color: #c8a84b; font-size: 11px; font-family: Georgia, serif;
    }
    .footer {
        text-align: center; color: #4a4a6a; font-size: 11px;
        margin-top: 40px; font-family: Georgia, serif; letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
WC2026_GROUPS = {
    "A": ["Mexico",        "South Africa",  "Korea Republic", "New Zealand"],
    "B": ["Canada",        "Switzerland",   "Qatar",          "Honduras"],
    "C": ["Brazil",        "Morocco",       "Haiti",          "Scotland"],
    "D": ["USA",           "Paraguay",      "Australia",      "Cameroon"],
    "E": ["Germany",       "Curacao",       "Ivory Coast",    "Ecuador"],
    "F": ["Netherlands",   "Japan",         "Tunisia",        "Senegal"],
    "G": ["Belgium",       "Egypt",         "Iran",           "Costa Rica"],
    "H": ["Spain",         "Cape Verde",    "Saudi Arabia",   "Uruguay"],
    "I": ["France",        "Algeria",       "Norway",         "Costa Rica"],
    "J": ["Argentina",     "Algeria",       "Austria",        "Jordan"],
    "K": ["Portugal",      "Colombia",      "Uzbekistan",     "Thailand"],
    "L": ["England",       "Croatia",       "Ghana",          "Panama"],
}
ALL_WC_TEAMS = list({t for grp in WC2026_GROUPS.values() for t in grp})

# Host nations get partial home advantage (is_neutral=0 for them)
HOST_NATIONS = {"USA", "Canada", "Mexico"}

CONFEDERATION = {
    "Spain":"UEFA","France":"UEFA","England":"UEFA","Germany":"UEFA",
    "Netherlands":"UEFA","Belgium":"UEFA","Portugal":"UEFA","Croatia":"UEFA",
    "Switzerland":"UEFA","Norway":"UEFA","Scotland":"UEFA","Austria":"UEFA",
    "Serbia":"UEFA","Denmark":"UEFA","Ukraine":"UEFA",
    "Brazil":"CONMEBOL","Argentina":"CONMEBOL","Colombia":"CONMEBOL",
    "Uruguay":"CONMEBOL","Ecuador":"CONMEBOL","Paraguay":"CONMEBOL","Chile":"CONMEBOL",
    "Morocco":"CAF","Senegal":"CAF","Egypt":"CAF","Ivory Coast":"CAF",
    "Ghana":"CAF","Cape Verde":"CAF","South Africa":"CAF","Tunisia":"CAF",
    "Algeria":"CAF","Cameroon":"CAF","Mali":"CAF",
    "Japan":"AFC","Korea Republic":"AFC","Iran":"AFC","Saudi Arabia":"AFC",
    "Australia":"AFC","Uzbekistan":"AFC","Jordan":"AFC","Qatar":"AFC","Thailand":"AFC",
    "Mexico":"CONCACAF","USA":"CONCACAF","Canada":"CONCACAF",
    "Panama":"CONCACAF","Curacao":"CONCACAF","Haiti":"CONCACAF","Honduras":"CONCACAF",
    "Costa Rica":"CONCACAF",
    "New Zealand":"OFC",
}
CONF_COLORS = {
    "UEFA":"#1a78cf","CONMEBOL":"#2ecc71","CAF":"#e67e22",
    "AFC":"#e74c3c","CONCACAF":"#27ae60","OFC":"#95a5a6",
}
FLAGS = {
    "Spain":"🇪🇸","France":"🇫🇷","Argentina":"🇦🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Mexico":"🇲🇽","Morocco":"🇲🇦","Japan":"🇯🇵","Portugal":"🇵🇹",
    "Netherlands":"🇳🇱","Senegal":"🇸🇳","Germany":"🇩🇪","USA":"🇺🇸",
    "Brazil":"🇧🇷","Australia":"🇦🇺","Croatia":"🇭🇷","Iran":"🇮🇷",
    "Belgium":"🇧🇪","Panama":"🇵🇦","Canada":"🇨🇦","Colombia":"🇨🇴",
    "Uruguay":"🇺🇾","Uzbekistan":"🇺🇿","Egypt":"🇪🇬","Switzerland":"🇨🇭",
    "Algeria":"🇩🇿","Qatar":"🇶🇦","Ecuador":"🇪🇨","Ivory Coast":"🇨🇮",
    "Norway":"🇳🇴","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Saudi Arabia":"🇸🇦","Jordan":"🇯🇴",
    "South Africa":"🇿🇦","Tunisia":"🇹🇳","Paraguay":"🇵🇾","Austria":"🇦🇹",
    "New Zealand":"🇳🇿","Haiti":"🇭🇹","Curacao":"🇨🇼","Ghana":"🇬🇭",
    "Cape Verde":"🇨🇻","Korea Republic":"🇰🇷","Honduras":"🇭🇳","Cameroon":"🇨🇲",
    "Costa Rica":"🇨🇷","Thailand":"🇹🇭",
}

def flag(team): return FLAGS.get(team, "🏳️")


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/best_model_v2.pkl", "rb") as f:
        return pickle.load(f)

saved    = load_model()
model    = saved["model"]
FEATURES = saved["features"]
base_df  = saved["df"].copy()
base_df["date"] = pd.to_datetime(base_df["date"])
elo_base = saved["elo_state"]
model_results = saved.get("model_results", {})
precomp_stats = saved.get("team_stats", {})
precomp_matchups = saved.get("matchups", {})
precomp_sim = saved.get("sim_results", {})


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "match_results" not in st.session_state:
    st.session_state.match_results = {}
if "sim_cache" not in st.session_state:
    st.session_state.sim_cache = None


# ─────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────
TW = {'FIFA World Cup':60,'UEFA Euro':50,'Copa America':50,
      'African Cup of Nations':50,'AFC Asian Cup':50,'Gold Cup':40,
      'CONCACAF Nations League':35,'UEFA Nations League':35,
      'UEFA Euro qualification':30,'FIFA World Cup qualification':30,
      'African Cup of Nations qualification':25,'AFC Asian Cup qualification':25}

def get_k(t):
    for key,k in TW.items():
        if key.lower() in t.lower(): return k
    return 25

@st.cache_data
def build_team_stats(_base_df, _elo_base, live_results_hash):
    """Rebuild team stats incorporating any live results."""
    elo = defaultdict(lambda: 1500.0, _elo_base)

    # Apply live results to Elo
    for key, res in st.session_state.match_results.items():
        home, away = key.split("|")
        he, ae = elo[home], elo[away]
        exp_h = 1/(1+10**((ae-he)/400))
        hs_, as__ = res["home_score"], res["away_score"]
        sh = 1.0 if hs_>as__ else (0.5 if hs_==as__ else 0.0)
        elo[home] = he + 60*(sh-exp_h)
        elo[away] = ae + 60*((1-sh)-(1-exp_h))

    # Build full df including live results
    df = _base_df.copy()
    if st.session_state.match_results:
        new_rows = []
        for key, res in st.session_state.match_results.items():
            home, away = key.split("|")
            hs_, as__ = res["home_score"], res["away_score"]
            result = 1 if hs_>as__ else (-1 if as__>hs_ else 0)
            new_rows.append({
                "home_team":home,"away_team":away,
                "home_score":hs_,"away_score":as__,"result":result,
                "date":pd.Timestamp("2026-06-15"),
                "home_ranking_pts":1500,"away_ranking_pts":1500,
                "tournament":"FIFA World Cup",
            })
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True).sort_values("date")

    stats = {}
    all_teams = set(df["home_team"])|set(df["away_team"])
    for team in all_teams:
        hm = df[df["home_team"]==team]; am = df[df["away_team"]==team]
        # form 5
        hw5 = list(hm.tail(5)["result"].map(lambda x:1 if x==1 else 0))
        aw5 = list(am.tail(5)["result"].map(lambda x:1 if x==-1 else 0))
        all5_dates = list(hm.tail(5)["date"])+list(am.tail(5)["date"])
        all5_wins  = hw5+aw5
        if all5_dates:
            combined5 = sorted(zip(all5_dates,all5_wins))[-5:]
            form5 = np.mean([w for _,w in combined5])
        else: form5=0.45
        # form 10
        hw10 = hm[["date"]].assign(w=hm["result"].map(lambda x:1 if x==1 else 0))
        aw10 = am[["date"]].assign(w=am["result"].map(lambda x:1 if x==-1 else 0))
        all10 = pd.concat([hw10,aw10]).sort_values("date").tail(10)
        form10 = all10["w"].mean() if len(all10) else 0.45
        # goals
        hm5=hm.tail(5); am5=am.tail(5)
        gs=list(hm5["home_score"])+list(am5["away_score"])
        gc=list(hm5["away_score"])+list(am5["home_score"])
        gs_mean=np.mean(gs) if gs else 1.2
        gc_mean=np.mean(gc) if gc else 1.0
        # ranking
        pts=1200.0
        if len(hm): pts=float(hm.iloc[-1]["home_ranking_pts"])
        elif len(am): pts=float(am.iloc[-1]["away_ranking_pts"])
        # confederation
        conf=CONFEDERATION.get(team,"UEFA")
        if len(hm) and "home_confederation" in hm.columns:
            v=hm.iloc[-1]["home_confederation"]
            if pd.notna(v): conf=v
        stats[team]={
            "elo":elo.get(team,1500.0),"pts":pts,
            "form5":form5,"form10":form10,
            "goals_scored5":gs_mean,"goals_conceded5":gc_mean,
            "confederation":conf,
        }
    return stats


def predict_match(home, away, team_stats, neutral=True):
    """Return (home_win%, draw%, away_win%) using the upgraded feature set."""
    hs  = team_stats.get(home, {"elo":1500,"pts":1200,"form5":0.45,"form10":0.45,
                                 "goals_scored5":1.2,"goals_conceded5":1.0,"confederation":"UEFA"})
    as_ = team_stats.get(away, {"elo":1500,"pts":1200,"form5":0.45,"form10":0.45,
                                 "goals_scored5":1.2,"goals_conceded5":1.0,"confederation":"UEFA"})
    past = base_df[(base_df["home_team"]==home)&(base_df["away_team"]==away)]
    h2h_hw = int((past["result"]==1).sum())
    h2h_d  = int((past["result"]==0).sum())
    h2h_hl = int((past["result"]==-1).sum())
    same_c = 1 if hs["confederation"]==as_["confederation"] else 0

    # Host nation advantage: not neutral for USA/Canada/Mexico
    is_neutral = 0 if (home in HOST_NATIONS and not neutral) else int(neutral)

    row = pd.DataFrame([{
        "elo_diff_w":            hs["elo"]-as_["elo"],
        "ranking_diff":          hs["pts"]-as_["pts"],
        "home_form_5":           hs["form5"],
        "away_form_5":           as_["form5"],
        "form_diff":             hs["form5"]-as_["form5"],
        "home_form_10":          hs["form10"],
        "away_form_10":          as_["form10"],
        "form_diff_10":          hs["form10"]-as_["form10"],
        "home_goals_scored_5":   hs["goals_scored5"],
        "home_goals_conceded_5": hs["goals_conceded5"],
        "away_goals_scored_5":   as_["goals_scored5"],
        "away_goals_conceded_5": as_["goals_conceded5"],
        "goal_diff_scored":      hs["goals_scored5"]-as_["goals_scored5"],
        "goal_diff_conceded":    as_["goals_conceded5"]-hs["goals_conceded5"],
        "h2h_home_wins":         h2h_hw,
        "h2h_draws":             h2h_d,
        "h2h_home_losses":       h2h_hl,
        "is_neutral":            is_neutral,
        "same_confederation":    same_c,
        "tournament_importance": 3,
    }])
    probs = model.predict_proba(row[FEATURES])[0].astype(float)
    probs /= probs.sum()
    # le classes order: check saved le
    le = saved["le"]
    classes = list(le.classes_)  # [-1, 0, 1]
    idx_hw = classes.index(1); idx_d = classes.index(0); idx_aw = classes.index(-1)
    return float(probs[idx_hw])*100, float(probs[idx_d])*100, float(probs[idx_aw])*100


def run_simulation(n, team_stats, groups):
    """Run N Monte Carlo simulations with proper tiebreakers and host advantage."""
    all_wc = [t for ts in groups.values() for t in ts]

    # Batch precompute all matchup probs
    rows, pairs = [], []
    for h in all_wc:
        for a in all_wc:
            if h==a: continue
            hs  = team_stats.get(h,{"elo":1500,"pts":1200,"form5":0.45,"form10":0.45,
                                    "goals_scored5":1.2,"goals_conceded5":1.0,"confederation":"UEFA"})
            as_ = team_stats.get(a,{"elo":1500,"pts":1200,"form5":0.45,"form10":0.45,
                                    "goals_scored5":1.2,"goals_conceded5":1.0,"confederation":"UEFA"})
            same_c=1 if hs["confederation"]==as_["confederation"] else 0
            is_n = 0 if h in HOST_NATIONS else 1
            rows.append({
                "elo_diff_w":hs["elo"]-as_["elo"],"ranking_diff":hs["pts"]-as_["pts"],
                "home_form_5":hs["form5"],"away_form_5":as_["form5"],
                "form_diff":hs["form5"]-as_["form5"],
                "home_form_10":hs["form10"],"away_form_10":as_["form10"],
                "form_diff_10":hs["form10"]-as_["form10"],
                "home_goals_scored_5":hs["goals_scored5"],"home_goals_conceded_5":hs["goals_conceded5"],
                "away_goals_scored_5":as_["goals_scored5"],"away_goals_conceded_5":as_["goals_conceded5"],
                "goal_diff_scored":hs["goals_scored5"]-as_["goals_scored5"],
                "goal_diff_conceded":as_["goals_conceded5"]-hs["goals_conceded5"],
                "h2h_home_wins":0,"h2h_draws":0,"h2h_home_losses":0,
                "is_neutral":is_n,"same_confederation":same_c,"tournament_importance":3,
            })
            pairs.append((h,a))

    batch = model.predict_proba(pd.DataFrame(rows)[FEATURES]).astype(float)
    batch /= batch.sum(axis=1,keepdims=True)
    le=saved["le"]; classes=list(le.classes_)
    idx_hw=classes.index(1); idx_d=classes.index(0); idx_aw=classes.index(-1)
    lookup={(h,a):(batch[i][idx_hw],batch[i][idx_d],batch[i][idx_aw]) for i,(h,a) in enumerate(pairs)}

    def sim_group(teams):
        pts={t:0 for t in teams}; gd={t:0 for t in teams}
        gf={t:0 for t in teams}; ga={t:0 for t in teams}
        h2h=defaultdict(lambda:defaultdict(int))  # h2h[team][pts/gd/gf]

        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                h,a=teams[i],teams[j]
                hw,d,aw=lookup.get((h,a),(0.4,0.2,0.4))
                r=random.random()
                if r<hw:
                    hs_=random.randint(1,3); as__=random.randint(0,hs_-1)
                    pts[h]+=3
                    h2h[h]["pts"]+=3
                elif r<hw+d:
                    hs_=random.randint(0,2); as__=hs_
                    pts[h]+=1; pts[a]+=1
                    h2h[h]["pts"]+=1; h2h[a]["pts"]+=1
                else:
                    as__=random.randint(1,3); hs_=random.randint(0,as__-1)
                    pts[a]+=3
                    h2h[a]["pts"]+=3
                gd[h]+=hs_-as__; gd[a]+=as__-hs_
                gf[h]+=hs_; gf[a]+=as__; ga[h]+=as__; ga[a]+=hs_
                h2h[h]["gd"]+=hs_-as__; h2h[a]["gd"]+=as__-hs_

        # Proper tiebreaker: pts → h2h pts → h2h gd → overall gd → gf → random
        def tiebreak_key(t):
            return (pts[t], h2h[t]["pts"], h2h[t]["gd"], gd[t], gf[t], random.random())
        ranked=sorted(teams,key=tiebreak_key,reverse=True)
        return ranked, pts, gd, gf

    def ko_winner(h,a):
        hw,d,aw=lookup.get((h,a),(0.4,0.2,0.4))
        p_h=hw+d/2; p_a=aw+d/2
        pr=np.array([p_h,p_a]); pr/=pr.sum()
        return h if random.random()<pr[0] else a

    wins=defaultdict(int); sf_c=defaultdict(int); f_c=defaultdict(int)

    for _ in range(n):
        qualifiers=[]; thirds=[]
        for _,teams in groups.items():
            ranked,pts,gd,gf=sim_group(teams)
            qualifiers.append(ranked[0]); qualifiers.append(ranked[1])
            thirds.append((ranked[2],pts[ranked[2]],gd[ranked[2]],gf[ranked[2]]))

        # Best 8 third-placed (12 groups × 3rd → pick top 8)
        thirds.sort(key=lambda x:(-x[1],-x[2],-x[3],random.random()))
        qualifiers+=[x[0] for x in thirds[:8]]
        random.shuffle(qualifiers)

        bracket=list(qualifiers)
        rnd=0
        while len(bracket)>1:
            if rnd==3:
                for t in bracket: sf_c[t]+=1
            if rnd==4:
                for t in bracket: f_c[t]+=1
            next_r=[]
            for i in range(0,len(bracket)-1,2):
                next_r.append(ko_winner(bracket[i],bracket[i+1]))
            bracket=next_r; rnd+=1
        wins[bracket[0]]+=1

    return wins,sf_c,f_c


# ─────────────────────────────────────────────────────────────
# LIVE STATS (with caching by results hash)
# ─────────────────────────────────────────────────────────────
live_hash = str(sorted(st.session_state.match_results.items()))
team_stats = build_team_stats(base_df, elo_base, live_hash)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏆 WC 2026 Predictor")
    st.markdown("---")
    st.markdown("**Model v2 — Upgrades**")
    st.markdown("""
- Weighted Elo (K by tournament)
- Form-5 & Form-10 windows
- Goals scored/conceded (last 5)
- Tournament importance feature
- Hyperparameter-tuned LR
- Proper WC tiebreaker rules
- Host-nation home advantage
    """)
    st.markdown("---")
    st.markdown("**Test Set Results (2022-2026)**")
    if model_results:
        for name,r in model_results.items():
            st.markdown(f"`{name}` — ll: {r['ll']:.4f}")
    st.markdown("---")
    st.markdown("**Live results update Elo.**  \nEnter actual scores in the  \nLive Results tab.")
    st.markdown("---")
    st.markdown('<div class="footer">Iyinoluwa Don-Taiwo · 2026</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
c1,c2=st.columns([3,1])
with c1:
    st.markdown("# 🏆 FIFA World Cup 2026")
    st.markdown("**ML Prediction System v2** — Tuned LR · Weighted Elo · Extended Form · Goals Features · Monte Carlo")
with c2:
    st.metric("Live Results", len(st.session_state.match_results))
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5=st.tabs([
    "⚽ Match Predictor",
    "🏟️ Knockout Simulator",
    "📊 Win Probabilities",
    "📋 Group Stage",
    "📡 Live Results",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Predict Any Match")
    st.markdown('<div class="info-box">Select any two World Cup teams. The upgraded model uses weighted Elo, form over 5 and 10 matches, goals scored and conceded, head-to-head history, and tournament importance to calculate real-time win probabilities.</div>', unsafe_allow_html=True)

    real_teams=sorted(ALL_WC_TEAMS)
    c1,c2,c3=st.columns([2,1,2])
    with c1:
        home=st.selectbox("🏠 Home / Team A", real_teams, index=real_teams.index("France") if "France" in real_teams else 0)
    with c2:
        st.markdown("<br><h3 style='text-align:center;color:#c8a84b;'>VS</h3>", unsafe_allow_html=True)
    with c3:
        default_idx=real_teams.index("Argentina") if "Argentina" in real_teams else 1
        away=st.selectbox("✈️ Away / Team B", real_teams, index=default_idx)

    neutral=st.checkbox("Neutral venue", value=True)

    if st.button("⚡ Predict Match", use_container_width=True):
        if home==away:
            st.warning("Select two different teams.")
        else:
            hw,d,aw_=predict_match(home,away,team_stats,neutral)

            if hw>aw_ and hw>d:     verdict=f"🟢 {home} are favoured to win"; vc="#1a6a4a"
            elif aw_>hw and aw_>d:  verdict=f"🔴 {away} are favoured to win"; vc="#6a1a1a"
            else:                    verdict="⚪ Match is too close to call — draw likely"; vc="#3a3a5a"

            st.markdown(f"""
            <div class="result-box">
                <h2 style="color:#f0e4c0;">{flag(home)} {home} &nbsp;vs&nbsp; {away} {flag(away)}</h2>
                <p style="color:{vc};font-size:16px;font-weight:bold;">{verdict}</p>
            </div>""", unsafe_allow_html=True)

            r1,r2,r3=st.columns(3)
            r1.metric(f"{home} Win", f"{hw:.1f}%")
            r2.metric("Draw", f"{d:.1f}%")
            r3.metric(f"{away} Win", f"{aw_:.1f}%")

            fig,ax=plt.subplots(figsize=(8,2.5))
            fig.patch.set_facecolor("#0d0d18"); ax.set_facecolor("#0d0d18")
            labels=[f"{home} Win","Draw",f"{away} Win"]; vals=[hw,d,aw_]
            colors=["#1a6a4a","#4a4a7a","#7a1a1a"]
            bars=ax.barh(labels,vals,color=colors,height=0.5,edgecolor="#1a1a2a")
            for bar,val in zip(bars,vals):
                ax.text(val+0.5,bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va="center",color="white",fontsize=11)
            ax.set_xlim(0,105); ax.set_xlabel("Probability (%)",color="#9a8a9a")
            ax.tick_params(colors="#9a8a9a"); ax.spines[:].set_visible(False)
            st.pyplot(fig); plt.close()

            st.markdown("#### Team Stats Comparison")
            hs_=team_stats.get(home,{}); as__=team_stats.get(away,{})
            comp=pd.DataFrame({
                "Stat":["Elo Rating","FIFA Points","Form (last 5)","Form (last 10)","Goals Scored/5","Goals Conceded/5","Confederation"],
                home: [f"{hs_.get('elo',1500):.0f}",f"{hs_.get('pts',1200):.0f}",
                        f"{hs_.get('form5',0.45)*100:.0f}%",f"{hs_.get('form10',0.45)*100:.0f}%",
                        f"{hs_.get('goals_scored5',1.2):.2f}",f"{hs_.get('goals_conceded5',1.0):.2f}",
                        CONFEDERATION.get(home,"?")],
                away: [f"{as__.get('elo',1500):.0f}",f"{as__.get('pts',1200):.0f}",
                        f"{as__.get('form5',0.45)*100:.0f}%",f"{as__.get('form10',0.45)*100:.0f}%",
                        f"{as__.get('goals_scored5',1.2):.2f}",f"{as__.get('goals_conceded5',1.0):.2f}",
                        CONFEDERATION.get(away,"?")],
            })
            st.dataframe(comp.set_index("Stat"),use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — KNOCKOUT SIMULATOR
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Knockout Bracket Simulator")
    st.markdown('<div class="info-box">Pick 16 teams for the Round of 16. The model predicts each match probabilistically using all 20 features — upsets happen naturally. Simulate repeatedly to see how the bracket plays out.</div>', unsafe_allow_html=True)

    real_wc=sorted(ALL_WC_TEAMS)
    default_r16=["France","Argentina","Spain","England","Brazil","Germany",
                 "Portugal","Netherlands","Morocco","Japan","Senegal","Croatia",
                 "Colombia","Mexico","Uruguay","Belgium"]
    default_r16=[t for t in default_r16 if t in real_wc][:16]

    selected=st.multiselect("Choose exactly 16 teams for Round of 16",real_wc,default=default_r16,max_selections=16)

    if len(selected)<2:
        st.info("Select at least 2 teams to simulate.")
    else:
        if st.button("🎯 Simulate Knockout",use_container_width=True):
            teams_ko=selected if len(selected)%2==0 else selected[:-1]
            current=list(teams_ko)
            round_names=["Round of 16","Quarterfinals","Semifinals","Final"]
            all_rounds={}; round_idx=0

            while len(current)>1:
                rname=round_names[min(round_idx,len(round_names)-1)]
                matchups=[]; winners=[]
                for i in range(0,len(current)-1,2):
                    h,a=current[i],current[i+1]
                    hw,d,aw_=predict_match(h,a,team_stats,neutral=True)
                    p_h=(hw+d/2)/100; p_a=(aw_+d/2)/100
                    pr=np.array([p_h,p_a]); pr/=pr.sum()
                    win=np.random.choice([h,a],p=pr)
                    matchups.append({"home":h,"away":a,"hw":round(hw,1),"d":round(d,1),"aw":round(aw_,1),"winner":win})
                    winners.append(win)
                all_rounds[rname]=matchups; current=winners; round_idx+=1

            champion=current[0]
            st.markdown(f"""
            <div class="winner-banner">
                <h1 style="color:#c8a84b;font-size:40px;">{flag(champion)}</h1>
                <h2 style="color:#c8a84b;">🏆 {champion} wins the World Cup!</h2>
            </div>""", unsafe_allow_html=True)

            for rname,matchups in all_rounds.items():
                st.markdown(f"#### {rname}")
                for m in matchups:
                    won_home=m["winner"]==m["home"]
                    ca,cb,cc=st.columns([3,2,3])
                    with ca:
                        w="bold" if won_home else "normal"
                        st.markdown(f"<p style='text-align:right;font-weight:{w};color:#e8e0d0;'>{flag(m['home'])} {m['home']}</p>",unsafe_allow_html=True)
                    with cb:
                        st.markdown(f"<p style='text-align:center;color:#c8a84b;font-size:12px;'>{m['hw']}% — {m['d']}% — {m['aw']}%</p>",unsafe_allow_html=True)
                    with cc:
                        w="bold" if not won_home else "normal"
                        st.markdown(f"<p style='font-weight:{w};color:#e8e0d0;'>{m['away']} {flag(m['away'])}</p>",unsafe_allow_html=True)
                    st.markdown(f"<p style='color:#c8a84b;font-size:12px;text-align:center;'>✓ {m['winner']} advances</p>",unsafe_allow_html=True)
                    st.markdown("---")


# ══════════════════════════════════════════════════════════════
# TAB 3 — WIN PROBABILITIES
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Tournament Win Probabilities")

    # Show precomputed results immediately
    if precomp_sim:
        st.markdown("#### Pre-Tournament Simulation (10,000 runs — from trained model)")
        top10=sorted(precomp_sim.items(),key=lambda x:-x[1]["win_pct"])[:10]
        mc1,mc2,mc3=st.columns(3)
        if top10: mc1.metric("🥇 "+top10[0][0],f"{top10[0][1]['win_pct']:.1f}%")
        if len(top10)>1: mc2.metric("🥈 "+top10[1][0],f"{top10[1][1]['win_pct']:.1f}%")
        if len(top10)>2: mc3.metric("🥉 "+top10[2][0],f"{top10[2][1]['win_pct']:.1f}%")

        table=[{"Rank":i+1,"Flag":flag(t),"Team":t,"Conf.":CONFEDERATION.get(t,"?"),
                "Win %":f"{d['win_pct']:.2f}%","Final %":f"{d['final_pct']:.1f}%","Semi %":f"{d['semi_pct']:.1f}%"}
               for i,(t,d) in enumerate(sorted(precomp_sim.items(),key=lambda x:-x[1]["win_pct"]))]
        st.dataframe(pd.DataFrame(table),use_container_width=True,hide_index=True)
        st.markdown("---")

    st.markdown("#### Run Live Simulation (uses current Elo + any live results)")
    cl,cr=st.columns([1,2])
    with cl:
        n_sims=st.select_slider("Simulations",options=[1000,2000,5000,10000],value=5000)
        run_btn=st.button("▶ Run Simulation",use_container_width=True)
    with cr:
        st.markdown('<div class="info-box">Runs the full 48-team bracket N times. Each match is sampled probabilistically — upsets happen. Win% = how often each team won across all runs. Reflects any live results you have entered.</div>',unsafe_allow_html=True)

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
        if len(ranked)>2: m3.metric("🥉 3rd Favourite",ranked[2][0],f"{ranked[2][1]/total*100:.1f}%")
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

        tbl=[{"Rank":i+1,"Flag":flag(t),"Team":t,"Conf.":CONFEDERATION.get(t,"?"),
              "Win %":f"{c/total*100:.2f}%","Semi %":f"{sf_c.get(t,0)/total*100:.1f}%","Final %":f"{f_c.get(t,0)/total*100:.1f}%"}
             for i,(t,c) in enumerate(ranked)]
        st.dataframe(pd.DataFrame(tbl),use_container_width=True,hide_index=True)
    else:
        st.info("Click **Run Simulation** to generate live probabilities.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — GROUP STAGE
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Group Stage Predictor")
    st.markdown('<div class="info-box">Select a group, simulate all 6 matches, and see predicted standings with proper World Cup tiebreakers (head-to-head points → head-to-head goal difference → overall goal difference → goals scored).</div>',unsafe_allow_html=True)

    sel_grp=st.selectbox("Select group",list(WC2026_GROUPS.keys()),format_func=lambda x:f"Group {x}")
    grp_teams=WC2026_GROUPS[sel_grp]

    st.markdown(f"#### Group {sel_grp} Teams")
    gc=st.columns(len(grp_teams))
    for i,team in enumerate(grp_teams):
        with gc[i]:
            conf=CONFEDERATION.get(team,"?"); color=CONF_COLORS.get(conf,"#555")
            st.markdown(f"""
            <div style="background:#12121e;border:1px solid #2a2040;border-top:3px solid {color};
                        border-radius:8px;padding:12px;text-align:center;">
                <div style="font-size:28px;">{flag(team)}</div>
                <div style="color:#e8e0d0;font-weight:bold;font-size:13px;">{team}</div>
                <div style="color:{color};font-size:11px;">{conf}</div>
            </div>""", unsafe_allow_html=True)

    if st.button("⚽ Simulate Group",use_container_width=True):
        pts={t:0 for t in grp_teams}; gd={t:0 for t in grp_teams}
        gf={t:0 for t in grp_teams}; ga={t:0 for t in grp_teams}
        h2h_pts=defaultdict(int); h2h_gd=defaultdict(int)
        results_log=[]

        for i in range(len(grp_teams)):
            for j in range(i+1,len(grp_teams)):
                h,a=grp_teams[i],grp_teams[j]
                hw,d,aw_=predict_match(h,a,team_stats,neutral=True)
                probs=np.array([hw/100,d/100,aw_/100]); probs/=probs.sum()
                o=np.random.choice(["Home Win","Draw","Away Win"],p=probs)
                if o=="Home Win":
                    hs_=random.randint(1,3); as__=random.randint(0,max(0,hs_-1))
                    pts[h]+=3; h2h_pts[h]+=3
                elif o=="Draw":
                    hs_=random.randint(0,2); as__=hs_
                    pts[h]+=1; pts[a]+=1; h2h_pts[h]+=1; h2h_pts[a]+=1
                else:
                    as__=random.randint(1,3); hs_=random.randint(0,max(0,as__-1))
                    pts[a]+=3; h2h_pts[a]+=3
                diff=hs_-as__
                gd[h]+=diff; gd[a]-=diff; gf[h]+=hs_; gf[a]+=as__
                ga[h]+=as__; ga[a]+=hs_; h2h_gd[h]+=diff; h2h_gd[a]-=diff
                results_log.append({
                    "Match":f"{flag(h)} {h} vs {a} {flag(a)}",
                    "Score":f"{hs_} - {as__}","Result":o,
                    f"{h} Win":f"{hw:.0f}%","Draw":f"{d:.0f}%",f"{a} Win":f"{aw_:.0f}%",
                })

        standings=sorted(grp_teams,key=lambda t:(pts[t],h2h_pts[t],h2h_gd[t],gd[t],gf[t],random.random()),reverse=True)

        st.markdown("#### Predicted Standings")
        for rank,team in enumerate(standings,1):
            qualifies=rank<=2
            badge="✅ Qualifies" if qualifies else "❌ Eliminated"
            bc="#1a4a1a" if qualifies else "#3a1a1a"; bb="#2a7a2a" if qualifies else "#7a2a2a"
            st.markdown(f"""
            <div style="background:{bc};border:1px solid {bb};border-radius:8px;
                        padding:10px 16px;margin:6px 0;display:flex;align-items:center;gap:12px;">
                <span style="font-size:20px;font-weight:bold;color:#c8a84b;">{rank}</span>
                <span style="font-size:22px;">{flag(team)}</span>
                <span style="flex:1;color:#e8e0d0;font-size:15px;">{team}</span>
                <span style="color:#9a8a9a;font-size:13px;">Pts:{pts[team]} GD:{gd[team]:+d} GF:{gf[team]}</span>
                <span style="font-size:12px;color:{'#5adb5a' if qualifies else '#db5a5a'};">{badge}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Match Results")
        st.dataframe(pd.DataFrame(results_log),use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — LIVE RESULTS
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Live Results Entry")
    st.markdown('<div class="info-box">Enter actual World Cup scores as matches finish. Each result recalculates the Elo ratings for both teams using the weighted K-factor (WC matches use K=60). This flows into every prediction and simulation automatically.</div>',unsafe_allow_html=True)

    live_teams=sorted(ALL_WC_TEAMS)
    with st.form("result_form"):
        st.markdown("#### Enter a Match Result")
        fc1,fc2,fc3=st.columns([3,1,3])
        with fc1: res_home=st.selectbox("Home Team",live_teams,key="res_home")
        with fc2: st.markdown("<br>",unsafe_allow_html=True)
        with fc3: res_away=st.selectbox("Away Team",live_teams,index=1,key="res_away")
        sc1,sc2=st.columns(2)
        with sc1: home_score=st.number_input("Home Score",min_value=0,max_value=20,value=1,step=1)
        with sc2: away_score=st.number_input("Away Score",min_value=0,max_value=20,value=0,step=1)
        submitted=st.form_submit_button("✅ Submit Result",use_container_width=True)
        if submitted:
            if res_home==res_away:
                st.error("Home and away teams must be different.")
            else:
                key=f"{res_home}|{res_away}"
                st.session_state.match_results[key]={"home_score":int(home_score),"away_score":int(away_score)}
                st.session_state.sim_cache=None
                st.success(f"✅ {res_home} {int(home_score)} – {int(away_score)} {res_away} recorded. Elo updated.")
                st.rerun()

    if st.session_state.match_results:
        st.markdown("#### Results Entered")
        for key,res in st.session_state.match_results.items():
            home,away=key.split("|"); hs_,as__=res["home_score"],res["away_score"]
            outcome=("🟢 "+home) if hs_>as__ else ("🔴 "+away) if as__>hs_ else "⚪ Draw"
            ca,cb=st.columns([3,2])
            ca.markdown(f"{flag(home)} **{home}** {hs_} – {as__} **{away}** {flag(away)}")
            cb.markdown(f"Winner: {outcome}")
        if st.button("🗑️ Clear All Results"):
            st.session_state.match_results={}; st.session_state.sim_cache=None; st.rerun()
    else:
        st.info("No results entered yet. Model is running on pre-tournament data.")

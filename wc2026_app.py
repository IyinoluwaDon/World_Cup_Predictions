"""
FIFA World Cup 2026 — ML Prediction App
========================================
Author  : Iyinoluwa Don-Taiwo · McPherson University
Model   : Logistic Regression trained on 11,437 competitive international matches
Features: Elo ratings, FIFA ranking points, recent form, head-to-head, venue context

Run locally : streamlit run wc2026_app.py
Deploy free : streamlit.io/cloud → connect GitHub repo
"""

import warnings
import pickle
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
    /* Dark background */
    .stApp { background-color: #0a0a12; }
    
    /* Headers */
    h1, h2, h3 { color: #f0e4c0 !important; font-family: Georgia, serif; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #12121e;
        border: 1px solid #2a2040;
        border-radius: 10px;
        padding: 12px;
    }
    [data-testid="metric-container"] label { color: #8a7a9a !important; font-size: 12px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #c8a84b !important; font-size: 26px; font-weight: bold; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0d0d18; border-right: 1px solid #1e1e2e; }
    [data-testid="stSidebar"] .stMarkdown { color: #9a8a9a; }

    /* Selectbox */
    .stSelectbox label { color: #9a8a9a !important; font-size: 13px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #0d0d18; border-bottom: 1px solid #2a2040; }
    .stTabs [data-baseweb="tab"] { color: #6a6a8a; font-family: Georgia, serif; letter-spacing: 0.05em; }
    .stTabs [aria-selected="true"] { color: #c8a84b !important; border-bottom: 2px solid #c8a84b !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #c8a84b, #a07830);
        color: #0a0a12;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        letter-spacing: 0.08em;
        font-family: Georgia, serif;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* Info boxes */
    .info-box {
        background: #12121e;
        border: 1px solid #2a2040;
        border-left: 4px solid #c8a84b;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        font-family: Georgia, serif;
        color: #ccc0b0;
        font-size: 14px;
    }
    .result-box {
        background: #0d1a2a;
        border: 1px solid #1a3a5a;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        text-align: center;
    }
    .prob-bar-container { margin: 8px 0; }
    .winner-banner {
        background: linear-gradient(135deg, #1a3a1a, #0d2a0d);
        border: 1px solid #2a6a2a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 12px 0;
    }
    .footer {
        text-align: center;
        color: #4a4a6a;
        font-size: 11px;
        margin-top: 40px;
        font-family: Georgia, serif;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
WC2026_GROUPS = {
    "A": ["Mexico",        "South Africa",  "Korea Republic", "TBD-A"],
    "B": ["Canada",        "TBD-B",         "Qatar",          "Switzerland"],
    "C": ["Brazil",        "Morocco",       "Haiti",          "Scotland"],
    "D": ["United States", "Paraguay",      "Australia",      "TBD-D"],
    "E": ["Germany",       "Curacao",       "Ivory Coast",    "Ecuador"],
    "F": ["Netherlands",   "Japan",         "TBD-F",          "Tunisia"],
    "G": ["Belgium",       "Egypt",         "Iran",           "New Zealand"],
    "H": ["Spain",         "Cabo Verde",    "Saudi Arabia",   "Uruguay"],
    "I": ["France",        "Senegal",       "TBD-I",          "Norway"],
    "J": ["Argentina",     "Algeria",       "Austria",        "Jordan"],
    "K": ["Portugal",      "TBD-K",         "Uzbekistan",     "Colombia"],
    "L": ["England",       "Croatia",       "Ghana",          "Panama"],
}
WC_TEAMS = [t for teams in WC2026_GROUPS.values() for t in teams]

CONFEDERATION = {
    "Spain":"UEFA","France":"UEFA","England":"UEFA","Germany":"UEFA",
    "Netherlands":"UEFA","Belgium":"UEFA","Portugal":"UEFA","Croatia":"UEFA",
    "Switzerland":"UEFA","Norway":"UEFA","Scotland":"UEFA","Austria":"UEFA",
    "Brazil":"CONMEBOL","Argentina":"CONMEBOL","Colombia":"CONMEBOL",
    "Uruguay":"CONMEBOL","Ecuador":"CONMEBOL","Paraguay":"CONMEBOL",
    "Morocco":"CAF","Senegal":"CAF","Egypt":"CAF","Ivory Coast":"CAF",
    "Ghana":"CAF","Cabo Verde":"CAF","South Africa":"CAF","Tunisia":"CAF",
    "Algeria":"CAF",
    "Japan":"AFC","Korea Republic":"AFC","Iran":"AFC","Saudi Arabia":"AFC",
    "Australia":"AFC","Uzbekistan":"AFC","Jordan":"AFC","Qatar":"AFC",
    "Mexico":"CONCACAF","United States":"CONCACAF","Canada":"CONCACAF",
    "Panama":"CONCACAF","Curacao":"CONCACAF","Haiti":"CONCACAF",
    "New Zealand":"OFC",
}
CONF_COLORS = {
    "UEFA":"#1a78cf","CONMEBOL":"#2ecc71","CAF":"#e67e22",
    "AFC":"#e74c3c","CONCACAF":"#27ae60","OFC":"#95a5a6",
}
FLAGS = {
    "Spain":"🇪🇸","France":"🇫🇷","Argentina":"🇦🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Mexico":"🇲🇽","Morocco":"🇲🇦","Japan":"🇯🇵","Portugal":"🇵🇹",
    "Netherlands":"🇳🇱","Senegal":"🇸🇳","Germany":"🇩🇪","United States":"🇺🇸",
    "Brazil":"🇧🇷","Australia":"🇦🇺","Croatia":"🇭🇷","Iran":"🇮🇷",
    "Belgium":"🇧🇪","Panama":"🇵🇦","Canada":"🇨🇦","Colombia":"🇨🇴",
    "Uruguay":"🇺🇾","Uzbekistan":"🇺🇿","Egypt":"🇪🇬","Switzerland":"🇨🇭",
    "Algeria":"🇩🇿","Qatar":"🇶🇦","Ecuador":"🇪🇨","Ivory Coast":"🇨🇮",
    "Norway":"🇳🇴","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Saudi Arabia":"🇸🇦","Jordan":"🇯🇴",
    "South Africa":"🇿🇦","Tunisia":"🇹🇳","Paraguay":"🇵🇾","Austria":"🇦🇹",
    "New Zealand":"🇳🇿","Haiti":"🇭🇹","Curacao":"🇨🇼","Ghana":"🇬🇭",
    "Cabo Verde":"🇨🇻","Korea Republic":"🇰🇷",
}
ELO_K = 40

def flag(team): return FLAGS.get(team, "🏳️")


# ─────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def build_team_stats(_df, _elo_state):
    """Precompute each team's current Elo, form and ranking points."""
    all_teams = set(_df["home_team"]) | set(_df["away_team"])
    stats = {}
    for team in all_teams:
        elo   = _elo_state.get(team, 1500.0)
        h_res = list(_df[_df["home_team"] == team]["result"].values)
        a_res = [({1:-1,-1:1,0:0}).get(int(r),0)
                 for r in _df[_df["away_team"] == team]["result"].values]
        all_r = h_res + a_res
        form  = float(np.mean([r==1 for r in all_r[-5:]])) if all_r else 0.5
        hr    = _df[_df["home_team"]==team]
        ar    = _df[_df["away_team"]==team]
        pts   = 1200.0
        if len(hr): pts = float(hr.iloc[-1]["home_ranking_pts"])
        elif len(ar): pts = float(ar.iloc[-1]["away_ranking_pts"])
        stats[team] = {"elo": elo, "form": form, "pts": pts}
    return stats


saved    = load_model()
model    = saved["model"]
FEATURES = saved["features"]
base_df  = saved["df"].copy()
base_df["date"] = pd.to_datetime(base_df["date"])


# ─────────────────────────────────────────────────────────────
# SESSION STATE  — live results storage
# ─────────────────────────────────────────────────────────────
# match_results stores real WC results entered by the user:
# key = "TeamA|TeamB", value = {"home_score": int, "away_score": int}
if "match_results" not in st.session_state:
    st.session_state.match_results = {}

if "sim_cache" not in st.session_state:
    st.session_state.sim_cache = None

if "custom_groups" not in st.session_state:
    st.session_state.custom_groups = {k: list(v) for k, v in WC2026_GROUPS.items()}


# ─────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────
def rebuild_elo_and_form(base_df, new_results):
    """
    Reconstruct Elo ratings and recent form by replaying the full
    match history PLUS any live results the user has entered.
    Returns updated team_stats dict.
    """
    # Start from base historical Elo state
    elo = defaultdict(lambda: 1500.0, saved["elo_state"])

    # Apply any live results entered by the user
    for key, res in new_results.items():
        home, away = key.split("|")
        he, ae = elo[home], elo[away]
        exp_h  = 1 / (1 + 10**((ae - he) / 400))
        hs, as_ = res["home_score"], res["away_score"]
        if hs > as_:   sh, sa = 1.0, 0.0
        elif hs == as_: sh, sa = 0.5, 0.5
        else:           sh, sa = 0.0, 1.0
        elo[home] = he + ELO_K * (sh - exp_h)
        elo[away] = ae + ELO_K * (sa - (1 - exp_h))

    # Rebuild form from base + new results
    df_live = base_df.copy()
    if new_results:
        new_rows = []
        for key, res in new_results.items():
            home, away = key.split("|")
            hs, as_ = res["home_score"], res["away_score"]
            result  = 1 if hs > as_ else (-1 if as_ > hs else 0)
            new_rows.append({
                "home_team": home, "away_team": away,
                "home_score": hs, "away_score": as_,
                "result": result,
                "date": pd.Timestamp("2026-06-15"),
                "home_ranking_pts": 1500, "away_ranking_pts": 1500,
            })
        df_live = pd.concat(
            [df_live, pd.DataFrame(new_rows)], ignore_index=True
        ).sort_values("date").reset_index(drop=True)

    return build_team_stats(df_live, dict(elo))


def predict_match(home, away, team_stats, neutral=True):
    """Return [away_win_prob, draw_prob, home_win_prob] for a given matchup."""
    hs  = team_stats.get(home, {"elo":1500,"form":0.5,"pts":1200})
    as_ = team_stats.get(away, {"elo":1500,"form":0.5,"pts":1200})
    row = pd.DataFrame([{
        "elo_diff":           hs["elo"]  - as_["elo"],
        "ranking_diff":       hs["pts"]  - as_["pts"],
        "home_form_5":        hs["form"],
        "away_form_5":        as_["form"],
        "form_diff":          hs["form"] - as_["form"],
        "h2h_home_wins":      0,
        "h2h_draws":          0,
        "h2h_home_losses":    0,
        "is_neutral":         int(neutral),
        "same_confederation": int(
            CONFEDERATION.get(home,"?") == CONFEDERATION.get(away,"?")
        ),
    }])
    p = model.predict_proba(row[FEATURES])[0].astype(float)
    p /= p.sum()
    return p  # [away_win, draw, home_win]


def run_simulation(n, team_stats, groups):
    """Run N Monte Carlo simulations of the full tournament."""
    # Precompute all matchup probs (batch is fast)
    all_wc = [t for ts in groups.values() for t in ts]
    rows, pairs = [], []
    for h in all_wc:
        for a in all_wc:
            if h == a: continue
            hs  = team_stats.get(h, {"elo":1500,"form":0.5,"pts":1200})
            as_ = team_stats.get(a, {"elo":1500,"form":0.5,"pts":1200})
            rows.append({
                "elo_diff":           hs["elo"]  - as_["elo"],
                "ranking_diff":       hs["pts"]  - as_["pts"],
                "home_form_5":        hs["form"],
                "away_form_5":        as_["form"],
                "form_diff":          hs["form"] - as_["form"],
                "h2h_home_wins":      0, "h2h_draws": 0, "h2h_home_losses": 0,
                "is_neutral":         1,
                "same_confederation": int(
                    CONFEDERATION.get(h,"?") == CONFEDERATION.get(a,"?")
                ),
            })
            pairs.append((h, a))

    batch = model.predict_proba(pd.DataFrame(rows)[FEATURES]).astype(float)
    batch /= batch.sum(axis=1, keepdims=True)
    lookup = {pair: batch[i] for i, pair in enumerate(pairs)}

    def gp(h, a): return lookup.get((h,a), np.array([0.33,0.34,0.33]))

    def sim_group(teams):
        pts = {t:0 for t in teams}; gd = {t:0 for t in teams}
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                h, a = teams[i], teams[j]
                o = np.random.choice([0,1,2], p=gp(h,a))
                if   o==2: pts[h]+=3; gd[h]+=1; gd[a]-=1
                elif o==1: pts[h]+=1; pts[a]+=1
                else:      pts[a]+=3; gd[a]+=1; gd[h]-=1
        return sorted(teams, key=lambda t:(pts[t],gd[t]),reverse=True), pts, gd

    def sim_ko(a, b):
        p = gp(a,b); hw=p[2]+p[1]*0.5; aw=p[0]+p[1]*0.5
        pr=np.array([hw,aw]); pr/=pr.sum()
        return np.random.choice([a,b],p=pr)

    def play(teams):
        return [sim_ko(teams[i],teams[i+1]) for i in range(0,len(teams)-1,2)]

    wins      = defaultdict(int)
    sf_count  = defaultdict(int)
    f_count   = defaultdict(int)
    np.random.seed(42)

    for _ in range(n):
        w, r, thirds = [], [], []
        for _, ts in groups.items():
            s, pts, gd = sim_group(ts)
            w.append(s[0]); r.append(s[1])
            thirds.append((s[2], pts[s[2]], gd[s[2]]))
        best8 = [t[0] for t in sorted(thirds,key=lambda x:(x[1],x[2]),reverse=True)[:8]]
        r32   = w + r + best8; np.random.shuffle(r32)
        r16   = play(r32)
        qf    = play(r16)
        sf    = play(qf)
        for t in sf: sf_count[t] += 1
        fi    = play(sf)
        for t in fi: f_count[t] += 1
        champ = play(fi)[0]
        wins[champ] += 1

    return wins, sf_count, f_count


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏆 WC 2026 Predictor")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("Logistic Regression  \n6,591 training matches  \nLog-loss: 0.8823")
    st.markdown("---")
    st.markdown("**Features used**")
    for feat in ["Elo rating difference","FIFA ranking points",
                 "Recent form (last 5)","Head-to-head record",
                 "Neutral venue flag","Confederation context"]:
        st.markdown(f"- {feat}")
    st.markdown("---")
    st.markdown("**Live results update Elo.**  \nEnter actual match scores  \nin the Live Results tab to  \nrecalculate all predictions.")
    st.markdown("---")
    st.markdown('<div class="footer">Iyinoluwa Don-Taiwo· 2026</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown("# 🏆 FIFA World Cup 2026")
    st.markdown("**ML Prediction System** — Logistic Regression · Elo Ratings · Monte Carlo Simulation")
with col_h2:
    n_results = len(st.session_state.match_results)
    st.metric("Live Results Entered", n_results)

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# REBUILD TEAM STATS (with any live results applied)
# ─────────────────────────────────────────────────────────────
team_stats = rebuild_elo_and_form(base_df, st.session_state.match_results)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
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
    st.markdown('<div class="info-box">Select any two teams. The model uses their current Elo rating, FIFA ranking points, and recent form to calculate win probabilities in real time.</div>',
                unsafe_allow_html=True)

    real_teams = [t for t in WC_TEAMS if not t.startswith("TBD")]

    c1, c2, c3 = st.columns([2,1,2])
    with c1:
        home = st.selectbox("🏠 Home / Team A", real_teams, index=real_teams.index("France"))
    with c2:
        st.markdown("<br><h3 style='text-align:center;color:#c8a84b;'>VS</h3>", unsafe_allow_html=True)
    with c3:
        default_away = real_teams.index("Argentina") if "Argentina" in real_teams else 1
        away = st.selectbox("✈️ Away / Team B", real_teams, index=default_away)

    neutral = st.checkbox("Neutral venue", value=True)

    if st.button("⚡ Predict Match", use_container_width=True):
        if home == away:
            st.warning("Please select two different teams.")
        else:
            p = predict_match(home, away, team_stats, neutral)
            away_win, draw, home_win = float(p[0])*100, float(p[1])*100, float(p[2])*100

            # Determine likely outcome
            if home_win > away_win and home_win > draw:
                verdict = f"🟢 {home} are favoured to win"
                verdict_color = "#1a6a4a"
            elif away_win > home_win and away_win > draw:
                verdict = f"🔴 {away} are favoured to win"
                verdict_color = "#6a1a1a"
            else:
                verdict = "⚪ Match is too close to call — draw likely"
                verdict_color = "#3a3a5a"

            st.markdown(f"""
            <div class="result-box">
                <h2 style="color:#f0e4c0;">{flag(home)} {home} &nbsp;vs&nbsp; {away} {flag(away)}</h2>
                <p style="color:{verdict_color};font-size:16px;font-weight:bold;">{verdict}</p>
            </div>
            """, unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            r1.metric(f"{home} Win", f"{home_win:.1f}%")
            r2.metric("Draw", f"{draw:.1f}%")
            r3.metric(f"{away} Win", f"{away_win:.1f}%")

            # Probability bars
            fig, ax = plt.subplots(figsize=(8,2.5))
            fig.patch.set_facecolor("#0d0d18")
            ax.set_facecolor("#0d0d18")
            labels = [f"{home} Win", "Draw", f"{away} Win"]
            values = [home_win, draw, away_win]
            colors = ["#1a6a4a","#4a4a7a","#7a1a1a"]
            bars   = ax.barh(labels, values, color=colors, height=0.5, edgecolor="#1a1a2a")
            for bar, val in zip(bars, values):
                ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}%", va="center", color="white", fontsize=11)
            ax.set_xlim(0,105); ax.set_xlabel("Probability (%)", color="#9a8a9a")
            ax.tick_params(colors="#9a8a9a"); ax.spines[:].set_visible(False)
            st.pyplot(fig); plt.close()

            # Team stat comparison
            st.markdown("#### Team Stats Comparison")
            hs = team_stats.get(home,{"elo":1500,"form":0.5,"pts":1200})
            as_ = team_stats.get(away,{"elo":1500,"form":0.5,"pts":1200})
            comp_df = pd.DataFrame({
                "Stat":      ["Elo Rating","FIFA Points","Recent Form (last 5)","Confederation"],
                home:        [f"{hs['elo']:.0f}", f"{hs['pts']:.0f}",
                              f"{hs['form']*100:.0f}% wins", CONFEDERATION.get(home,"?")],
                away:        [f"{as_['elo']:.0f}", f"{as_['pts']:.0f}",
                              f"{as_['form']*100:.0f}% wins", CONFEDERATION.get(away,"?")],
            })
            st.dataframe(comp_df.set_index("Stat"), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — KNOCKOUT SIMULATOR
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Knockout Bracket Simulator")
    st.markdown('<div class="info-box">Pick the winner of each match yourself. The model predicts probabilities — you decide who advances. Track your bracket all the way to the final.</div>',
                unsafe_allow_html=True)

    real_wc = [t for t in WC_TEAMS if not t.startswith("TBD")]

    st.markdown("#### Build Your Round of 16")
    st.markdown("Select 16 teams to simulate the full knockout bracket:")

    # Let user pick 16 teams for the R16
    default_r16 = ["France","Argentina","Spain","England","Brazil","Germany",
                   "Portugal","Netherlands","Morocco","Japan","Senegal","Croatia",
                   "Colombia","Mexico","Uruguay","Belgium"]
    default_r16 = [t for t in default_r16 if t in real_wc]

    selected = st.multiselect(
        "Choose exactly 16 teams for Round of 16",
        real_wc,
        default=default_r16[:16],
        max_selections=16,
    )

    if len(selected) < 2:
        st.info("Select at least 2 teams to simulate.")
    else:
        if st.button("🎯 Simulate Knockout", use_container_width=True):
            # Pair them up
            teams_ko = selected if len(selected) % 2 == 0 else selected[:-1]
            rounds = {"Round of 16": [], "Quarterfinals": [],
                      "Semifinals": [], "Final": []}

            current = teams_ko
            round_names = ["Round of 16","Quarterfinals","Semifinals","Final"]
            all_rounds  = {}
            round_idx   = 0

            while len(current) > 1:
                rname    = round_names[min(round_idx, len(round_names)-1)]
                matchups = []
                winners  = []
                for i in range(0, len(current)-1, 2):
                    h, a = current[i], current[i+1]
                    p    = predict_match(h, a, team_stats, neutral=True)
                    hw   = float(p[2])+float(p[1])*0.5
                    aw   = float(p[0])+float(p[1])*0.5
                    pr   = np.array([hw,aw]); pr /= pr.sum()
                    win  = np.random.choice([h,a], p=pr)
                    matchups.append({
                        "home": h, "away": a,
                        "home_win_pct": round(float(p[2])*100,1),
                        "draw_pct":     round(float(p[1])*100,1),
                        "away_win_pct": round(float(p[0])*100,1),
                        "winner": win,
                    })
                    winners.append(win)
                all_rounds[rname] = matchups
                current   = winners
                round_idx += 1

            champion = current[0]

            st.markdown(f"""
            <div class="winner-banner">
                <h1 style="color:#c8a84b;font-size:40px;">{flag(champion)}</h1>
                <h2 style="color:#c8a84b;">🏆 {champion} wins the World Cup!</h2>
            </div>
            """, unsafe_allow_html=True)

            for rname, matchups in all_rounds.items():
                st.markdown(f"#### {rname}")
                for m in matchups:
                    won_home = m["winner"] == m["home"]
                    col_a, col_b, col_c = st.columns([3,2,3])
                    with col_a:
                        weight = "bold" if won_home else "normal"
                        st.markdown(f"<p style='text-align:right;font-weight:{weight};color:#e8e0d0;'>{flag(m['home'])} {m['home']}</p>",
                                    unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"<p style='text-align:center;color:#c8a84b;font-size:12px;'>{m['home_win_pct']}% — {m['draw_pct']}% — {m['away_win_pct']}%</p>",
                                    unsafe_allow_html=True)
                    with col_c:
                        weight = "bold" if not won_home else "normal"
                        st.markdown(f"<p style='font-weight:{weight};color:#e8e0d0;'>{m['away']} {flag(m['away'])}</p>",
                                    unsafe_allow_html=True)
                    winner_style = "color:#c8a84b;font-size:12px;text-align:center;"
                    st.markdown(f"<p style='{winner_style}'>✓ {m['winner']} advances</p>",
                                unsafe_allow_html=True)
                    st.markdown("---")


# ══════════════════════════════════════════════════════════════
# TAB 3 — WIN PROBABILITIES
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Tournament Win Probabilities")

    c_left, c_right = st.columns([1,2])
    with c_left:
        n_sims = st.select_slider(
            "Number of simulations",
            options=[1000, 2000, 5000, 10000],
            value=5000,
        )
        run_btn = st.button("▶ Run Simulation", use_container_width=True)

    with c_right:
        st.markdown('<div class="info-box">The model simulates the entire 48-team tournament thousands of times. Each run samples match outcomes probabilistically — upsets happen naturally. The win probability for each team is how often they won across all runs.</div>',
                    unsafe_allow_html=True)

    if run_btn or st.session_state.sim_cache is not None:
        if run_btn:
            with st.spinner(f"Running {n_sims:,} simulations..."):
                wins, sf_c, f_c = run_simulation(
                    n_sims, team_stats, st.session_state.custom_groups
                )
                st.session_state.sim_cache = (wins, sf_c, f_c, n_sims)
        else:
            wins, sf_c, f_c, n_sims = st.session_state.sim_cache

        ranked = sorted(wins.items(), key=lambda x: -x[1])
        total  = sum(wins.values())

        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        if ranked:
            m1.metric("🥇 Predicted Champion", ranked[0][0], f"{ranked[0][1]/total*100:.1f}%")
        if len(ranked)>1:
            m2.metric("🥈 Runner Up",           ranked[1][0], f"{ranked[1][1]/total*100:.1f}%")
        if len(ranked)>2:
            m3.metric("🥉 Third Favourite",     ranked[2][0], f"{ranked[2][1]/total*100:.1f}%")
        m4.metric("Simulations Run", f"{total:,}")

        st.markdown("---")

        # Full chart
        fig, ax = plt.subplots(figsize=(10, max(8, len(ranked)*0.38)))
        fig.patch.set_facecolor("#0a0a12")
        ax.set_facecolor("#0a0a12")

        t_names = [r[0] for r in ranked]
        t_probs = [r[1]/total*100 for r in ranked]
        t_cols  = [CONF_COLORS.get(CONFEDERATION.get(t,"?"),"#888") for t in t_names]

        bars = ax.barh(t_names, t_probs, color=t_cols, edgecolor="#1a1a2a", linewidth=0.5)
        ax.axvline(100/48, color="#c8a84b", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"Random baseline ({100/48:.1f}%)")

        for bar, val in zip(bars, t_probs):
            if val >= 0.3:
                ax.text(val+0.1, bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}%", va="center", color="white", fontsize=8)

        ax.set_xlabel("Win Probability (%)", color="#9a8a9a")
        ax.set_title(f"World Cup 2026 — Win Probabilities ({total:,} simulations)",
                     color="#f0e4c0", fontsize=12, fontweight="bold")
        ax.tick_params(colors="#9a8a9a"); ax.spines[:].set_visible(False)

        # Confederation legend
        patches = [mpatches.Patch(color=v, label=k) for k,v in CONF_COLORS.items()]
        ax.legend(handles=patches + [ax.lines[0]], loc="lower right",
                  facecolor="#0d0d18", labelcolor="white", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Data table
        st.markdown("#### Full Results Table")
        table_data = [{
            "Rank":       i+1,
            "Flag":       flag(t),
            "Team":       t,
            "Conf.":      CONFEDERATION.get(t,"?"),
            "Win %":      f"{c/total*100:.2f}%",
            "Semis %":    f"{sf_c.get(t,0)/total*100:.1f}%",
            "Final %":    f"{f_c.get(t,0)/total*100:.1f}%",
        } for i,(t,c) in enumerate(ranked)]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    else:
        st.info("Click **Run Simulation** to generate win probabilities.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — GROUP STAGE
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Group Stage Predictor")
    st.markdown('<div class="info-box">Select a group to simulate all 6 matches and predict the standings. Run multiple times — probabilistic sampling means results vary naturally.</div>',
                unsafe_allow_html=True)

    selected_group = st.selectbox("Select group", list(WC2026_GROUPS.keys()),
                                  format_func=lambda x: f"Group {x}")
    group_teams = WC2026_GROUPS[selected_group]
    real_group  = [t for t in group_teams if not t.startswith("TBD")]

    # Show group teams
    st.markdown(f"#### Group {selected_group} Teams")
    gc = st.columns(len(group_teams))
    for i, team in enumerate(group_teams):
        with gc[i]:
            conf  = CONFEDERATION.get(team, "?")
            color = CONF_COLORS.get(conf, "#555")
            is_tbd = team.startswith("TBD")
            st.markdown(f"""
            <div style="background:#12121e;border:1px solid #2a2040;border-top:3px solid {color};
                        border-radius:8px;padding:12px;text-align:center;">
                <div style="font-size:28px;">{flag(team)}</div>
                <div style="color:#e8e0d0;font-weight:bold;font-size:13px;">{team}</div>
                <div style="color:{color};font-size:11px;">{conf}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("⚽ Simulate Group", use_container_width=True) and len(real_group) >= 2:
        pts  = {t:0 for t in real_group}
        gd   = {t:0 for t in real_group}
        gf   = {t:0 for t in real_group}
        results_log = []

        for i in range(len(real_group)):
            for j in range(i+1, len(real_group)):
                home, away = real_group[i], real_group[j]
                p    = predict_match(home, away, team_stats, neutral=True)
                probs= np.array([float(p[2]),float(p[1]),float(p[0])])
                o    = np.random.choice(["Home Win","Draw","Away Win"], p=probs)
                if o == "Home Win":
                    pts[home]+=3; gd[home]+=1; gd[away]-=1
                    gf[home]+=2; gf[away]+=1
                    score = "2 - 1"
                elif o == "Draw":
                    pts[home]+=1; pts[away]+=1
                    gf[home]+=1; gf[away]+=1
                    score = "1 - 1"
                else:
                    pts[away]+=3; gd[away]+=1; gd[home]-=1
                    gf[away]+=2; gf[home]+=1
                    score = "1 - 2"
                results_log.append({
                    "Match": f"{flag(home)} {home} vs {away} {flag(away)}",
                    "Score": score, "Result": o,
                    f"{home} Win": f"{float(p[2])*100:.0f}%",
                    "Draw":        f"{float(p[1])*100:.0f}%",
                    f"{away} Win": f"{float(p[0])*100:.0f}%",
                })

        standings = sorted(real_group,
                           key=lambda t:(pts[t],gd[t],gf[t]), reverse=True)

        st.markdown("#### Simulated Standings")
        for rank, team in enumerate(standings, 1):
            qualifier = rank <= 2
            badge = "✅ Qualifies" if qualifier else "❌ Eliminated"
            b_color = "#1a4a1a" if qualifier else "#3a1a1a"
            b_border = "#2a7a2a" if qualifier else "#7a2a2a"
            st.markdown(f"""
            <div style="background:{b_color};border:1px solid {b_border};
                        border-radius:8px;padding:10px 16px;margin:6px 0;
                        display:flex;align-items:center;gap:12px;">
                <span style="font-size:20px;font-weight:bold;color:#c8a84b;">{rank}</span>
                <span style="font-size:22px;">{flag(team)}</span>
                <span style="flex:1;color:#e8e0d0;font-size:15px;">{team}</span>
                <span style="color:#9a8a9a;font-size:13px;">Pts: {pts[team]} | GD: {gd[team]:+d}</span>
                <span style="font-size:12px;color:{'#5adb5a' if qualifier else '#db5a5a'};">{badge}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Match Results")
        log_df = pd.DataFrame(results_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — LIVE RESULTS
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Live Results Entry")
    st.markdown('<div class="info-box">Enter actual World Cup scores as matches finish. Every result you enter recalculates the Elo ratings for both teams — which updates all predictions and simulations automatically. This is what makes the model live.</div>',
                unsafe_allow_html=True)

    real_teams_live = [t for t in WC_TEAMS if not t.startswith("TBD")]

    with st.form("result_form"):
        st.markdown("#### Enter a Match Result")
        fc1, fc2, fc3 = st.columns([3,1,3])
        with fc1:
            res_home = st.selectbox("Home Team", real_teams_live, key="res_home")
        with fc2:
            st.markdown("<br>", unsafe_allow_html=True)
        with fc3:
            res_away = st.selectbox("Away Team", real_teams_live,
                                    index=1, key="res_away")
        sc1, sc2 = st.columns(2)
        with sc1:
            home_score = st.number_input("Home Score", min_value=0, max_value=20,
                                         value=1, step=1)
        with sc2:
            away_score = st.number_input("Away Score", min_value=0, max_value=20,
                                         value=0, step=1)
        submitted = st.form_submit_button("✅ Submit Result", use_container_width=True)

        if submitted:
            if res_home == res_away:
                st.error("Home and away teams must be different.")
            else:
                key = f"{res_home}|{res_away}"
                st.session_state.match_results[key] = {
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                }
                st.session_state.sim_cache = None  # invalidate simulation cache
                st.success(f"✅ {res_home} {int(home_score)} – {int(away_score)} {res_away} recorded. Elo updated.")
                st.rerun()

    if st.session_state.match_results:
        st.markdown("#### Results Entered So Far")
        for key, res in st.session_state.match_results.items():
            home, away = key.split("|")
            hs, as_ = res["home_score"], res["away_score"]
            outcome = ("🟢 " + home) if hs > as_ else ("🔴 " + away) if as_ > hs else "⚪ Draw"
            col_a, col_b, col_c = st.columns([3,2,2])
            col_a.markdown(f"{flag(home)} **{home}** {hs} – {as_} **{away}** {flag(away)}")
            col_b.markdown(f"Winner: {outcome}")
        if st.button("🗑️ Clear All Results", use_container_width=False):
            st.session_state.match_results = {}
            st.session_state.sim_cache     = None
            st.rerun()
    else:
        st.info("No results entered yet. The model is running on pre-tournament data.")

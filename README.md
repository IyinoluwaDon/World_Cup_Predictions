# 🏆 FIFA World Cup 2026 — ML Prediction System

**Predicting match outcomes and simulating the 2026 FIFA World Cup using machine learning.**

Built by [Iyinoluwa Don-Taiwo](https://www.instagram.com/donlovesml_/)

---

## Overview

This project builds an end-to-end machine learning system that predicts the outcomes of international football matches and simulates the entire 2026 FIFA World Cup tournament. The system is trained on 11,437 competitive international matches spanning 2002 to 2026, uses Elo ratings computed from scratch alongside FIFA ranking data, and runs 10,000 Monte Carlo simulations of the 48-team bracket to produce win probabilities for every competing nation.

The accompanying Streamlit web app allows fully interactive predictions — including live result entry that recalculates Elo ratings and updates all predictions in real time as the tournament progresses.

---

## Live Demo

```bash
streamlit run wc2026_app.py
```

---

## Project Structure

```
wc2026-predictor/
│
├── wc2026_app.py              # Streamlit web application (5 interactive tabs)
├── wc2026_pipeline.py         # Full ML pipeline — EDA, features, training, simulation
├── best_model.pkl             # Trained model + Elo state + feature matrix
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   ├── raw/
│   │   ├── results.csv                    # International match results (martj42, Kaggle)
│   │   └── fifa_ranking-2024-06-20.csv    # FIFA world rankings (cashncarry, Kaggle)
│   └── processed/
│       ├── master_matches.csv             # Merged, cleaned match dataset
│       └── feature_matrix.csv            # Final feature matrix used for training
│
├── models/
│   └── best_model.pkl                     # Saved model (also in root for deployment)
│
└── outputs/
    ├── eda_overview.png                   # EDA charts
    ├── feature_importance.png             # Feature importance bar chart
    ├── win_probabilities.png              # Full 48-team win probability chart
    ├── top10_chart.png                    # Top 10 teams chart
    └── simulation_results.csv            # Monte Carlo output table
```

---

## The Problem

**Task:** Multiclass classification — predict the outcome of an international football match as Home Win / Draw / Away Win from the home team's perspective.

**Encoding:** `1` = Home Win, `0` = Draw, `−1` = Away Win

**Ultimate goal:** Use the trained classifier inside a Monte Carlo simulation to assign win probabilities to all 48 teams in the 2026 World Cup bracket.

**Primary metric:** Log-loss (cross-entropy). This rewards well-calibrated probability estimates rather than just correct labels — essential for a simulation system that needs reliable probabilities, not just predictions.

---

## Datasets

| # | Dataset | Source | Used For |
|---|---------|--------|----------|
| 1 | International Football Results 1872–present | [Kaggle / martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Match outcomes, Elo computation, form windows, H2H |
| 2 | FIFA World Rankings 1992–2024 | [Kaggle / cashncarry](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Team strength feature — ranking points over time |

### Data decisions

**Why only two datasets?** A third dataset (FIFA 21 Player Ratings) was evaluated and dropped — it is a static 2020/2021 snapshot, covers club ratings rather than national team context, and is missing players who emerged after 2021 (e.g. Lamine Yamal). Adding stale, misaligned data introduces noise rather than signal. FIFA Rankings serve as a superior, time-aligned proxy for team strength.

**Why competitive matches only?** Friendlies are played with rotated squads and low stakes. Including them reduces signal quality. The filtered dataset keeps: FIFA World Cup, World Cup qualification, UEFA Euro, UEFA Nations League, Copa América, AFCON, AFC Asian Cup, Gold Cup, CONCACAF Nations League, and FIFA Confederations Cup.

**Why 2002 onward?** Football before this era is structurally too different — tactics, fitness levels, global participation, and data quality all changed substantially. Post-2002 matches are more predictive of 2026 outcomes.

---

## Data Pipeline

### Step 1 — Team name normalisation

The two datasets use inconsistent naming conventions. A name map was built and applied before any merging:

```python
name_map = {
    'Czech Republic':   'Czechia',
    'DR Congo':         'Congo DR',
    'Cape Verde':       'Cabo Verde',
    'Curaçao':          'Curacao',
    'North Macedonia':  'Macedonia',
}
```

### Step 2 — Date-aligned ranking merge

FIFA publishes rankings on a monthly schedule, not daily. Match dates and ranking dates never align exactly. `pandas.merge_asof()` was used to find the most recent ranking published *before or on* each match date — for both the home and away team simultaneously.

```python
merged = pd.merge_asof(
    results.sort_values('date'),
    rankings.sort_values('rank_date'),
    left_on='date',
    right_on='rank_date',
    left_by='home_team',
    right_by='country_full'
)
```

This produces a single `master_matches.csv` — one row per match, with ranking columns from the correct historical snapshot.

### Step 3 — Missing value handling

Matches after the rankings cutoff (June 2024) had no ranking data. These were filled with each team's last known ranking using forward-fill — a reasonable approximation for the 1,929 affected rows.

**Final dataset:** 11,437 rows · 14 columns · 2002–2026

---

## Feature Engineering

Ten features were engineered from raw match data. All are computed using only past data — never the current or future match — eliminating data leakage.

| Feature | Description | How Computed |
|---------|-------------|--------------|
| `elo_diff` | Relative historical strength | `home_elo − away_elo` (from scratch, chronological loop) |
| `ranking_diff` | FIFA points gap at match time | `home_ranking_pts − away_ranking_pts` (merge_asof) |
| `home_form_5` | Home team win rate in last 5 matches | Rolling window, date-filtered, leak-free |
| `away_form_5` | Away team win rate in last 5 matches | Same method |
| `form_diff` | Net momentum difference | `home_form_5 − away_form_5` |
| `h2h_home_wins` | Home team head-to-head wins vs this opponent | Count of past meetings |
| `h2h_draws` | Head-to-head draws | Count of past meetings |
| `h2h_home_losses` | Home team head-to-head losses vs this opponent | Count of past meetings |
| `is_neutral` | Neutral venue flag | `1` = neutral, `0` = home advantage |
| `same_confederation` | Both teams from same confederation | `1` = same, `0` = cross-confederation |

### Elo ratings

Elo is a dynamic rating system that updates after every match. Every team starts at 1500. The winner gains points and the loser loses them — the amount transferred depends on how surprising the result was. A heavy underdog winning transfers more points than a favourite winning.

```python
K = 40

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

# Update after each match, in chronological order
elo[home] = home_elo + K * (actual_home - expected_home)
elo[away] = away_elo + K * (actual_away - expected_away)
```

The Elo difference (`elo_diff`) is consistently the strongest single predictor in the model.

---

## Model Building

### Train / Validate / Test Split

Data was split by time, not randomly. Random splits cause data leakage in time-series data — future matches would appear in training, inflating metrics artificially.

| Split | Date Range | Matches |
|-------|-----------|---------|
| Training | Before 2018 | 6,591 |
| Test | 2022 onward | 2,718 |

### Model Comparison

Four models were trained and evaluated. The primary metric is log-loss — lower means better-calibrated probabilities.

| Model | Accuracy | Log-Loss |
|-------|----------|----------|
| **Logistic Regression** | **0.5982** | **0.8823** ✓ |
| LightGBM | 0.5791 | 0.9267 |
| Random Forest | 0.5717 | 0.9696 |
| XGBoost | 0.5640 | 1.0055 |

**Logistic Regression won.** This is not surprising — it is a well-understood result in sports prediction literature. The features are doing the heavy lifting. A simpler model with good features outperforms complex models with the same features on tabular sports data, because the underlying signal is inherently noisy.

### Classification Report (Logistic Regression on test set)

```
              precision    recall    f1-score    support

   Away Win       0.59      0.62        0.61        850
       Draw       0.25      0.04        0.07        609
   Home Win       0.62      0.85        0.72       1259

   accuracy                             0.60       2718
```

Draw prediction is the weakest class — F1 of 0.07. This is a known challenge in football modelling. Draws are the hardest outcome to predict in any sport because they represent a genuine balance of strength, not a directional result. The model correctly learns that draws are less likely than decisive outcomes but struggles to identify exactly when they happen.

---

## Tournament Simulation

### Approach

Rather than predicting a single tournament outcome (which would be overconfident), the model runs 10,000 full tournament simulations. Each simulation:

1. Simulates all 12 groups using probabilistic match sampling
2. Advances the top 2 from each group + 8 best third-place teams to the Round of 32
3. Runs four knockout rounds to a champion
4. Records the winner

After 10,000 runs, each team's win probability is their win count divided by 10,000.

### Why probabilistic sampling?

Instead of always picking the most likely outcome, each match result is sampled from the predicted probability distribution:

```python
outcome = np.random.choice(['home_win', 'draw', 'away_win'], p=[p_hw, p_draw, p_aw])
```

This means upsets happen naturally — a team with 15% win probability still wins in roughly 1 in 7 simulations. The Monte Carlo distribution captures genuine uncertainty rather than a single deterministic path.

### Knockout draw handling

There are no draws in real knockout football. When the model predicts draw probability in a knockout match, it is redistributed equally between the two teams to simulate extra time and penalties:

```python
home_knockout_prob = p_home_win + p_draw * 0.5
away_knockout_prob = p_away_win + p_draw * 0.5
```

### Performance optimisation

Running 600,000 individual model predictions (10,000 simulations × ~60 matches) inside a Python loop is too slow. All pairwise matchup probabilities for the 48 WC teams are precomputed in a single vectorised batch call before the simulation loop begins:

```python
batch_probs = model.predict_proba(all_matchup_features)   # 2,256 predictions at once
prob_lookup = {pair: batch_probs[i] for i, pair in enumerate(pairs)}
```

The simulation then uses only dictionary lookups — reducing 10,000-run simulation time from minutes to seconds.

### Results (10,000 simulations)

| Rank | Team | Win Probability |
|------|------|----------------|
| 1 | 🇪🇸 Spain | 13.45% |
| 2 | 🇫🇷 France | 9.62% |
| 3 | 🇦🇷 Argentina | 8.08% |
| 4 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 7.42% |
| 5 | 🇲🇽 Mexico | 5.90% |
| 6 | 🇲🇦 Morocco | 5.52% |
| 7 | 🇯🇵 Japan | 4.90% |
| 8 | 🇵🇹 Portugal | 4.44% |
| 9 | 🇳🇱 Netherlands | 3.99% |
| 10 | 🇸🇳 Senegal | 3.71% |

Random baseline (equal probability for all 48 teams): **2.08%**

---

## Streamlit App — Feature Guide

The web app has five fully interactive tabs.

### ⚽ Match Predictor

Select any two World Cup teams. The model looks up their current Elo rating, FIFA ranking points, and recent form to predict win/draw/loss probabilities in real time. Includes a neutral venue toggle and a side-by-side stat comparison table.

### 🏟️ Knockout Simulator

Choose 16 teams for the Round of 16 and simulate the full knockout bracket. The model predicts probabilities for each match and samples the winner probabilistically — run it multiple times and the bracket changes. Follows each team all the way to the final.

### 📊 Win Probabilities

Run Monte Carlo simulations on demand — 1,000, 2,000, 5,000, or 10,000 runs. Generates the full ranked win probability table with Semifinals reach % and Finals reach % per team. Automatically reruns when live results are entered.

### 📋 Group Stage

Pick any of the 12 groups and simulate all 6 round-robin matches. Displays predicted standings with points, goal difference, and qualifier status for each team.

### 📡 Live Results

Enter actual World Cup match scores as they happen. Each result immediately recalculates the Elo ratings for both teams — which propagates into every prediction and simulation in the app. This is what makes the model live throughout the tournament.

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/wc2026-predictor
cd wc2026-predictor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run wc2026_app.py
```

The app opens at `http://localhost:8501`

To retrain the model from scratch:

```bash
# Place master_matches.csv in data/processed/
python wc2026_pipeline.py
```

---

## Deploy to Streamlit Community Cloud

```
1. Push wc2026_app.py, best_model.pkl, and requirements.txt to a GitHub repo
2. Go to share.streamlit.io
3. Sign in with GitHub
4. New app → select your repo → set main file to wc2026_app.py
5. Deploy
```

Free tier. No credit card. Deploys in under two minutes. The URL is shareable immediately.

---

## Limitations

**Draws are hard.** The model has an F1 of 0.07 on draw prediction. This is a known limitation of all football prediction models — draws are genuinely difficult to predict because they depend on match-level dynamics (pace of play, substitutions, red cards) that no pre-match model can see.

**Rankings cutoff.** FIFA rankings data cuts off at June 2024. Matches from June 2024 to March 2026 use each team's last known ranking via forward-fill. This is a reasonable approximation but not ideal.

**Squad changes.** The model does not know about injuries, suspensions, or manager changes. A star player's absence the day before a match is invisible to the model.

**TBD playoff teams.** Four spots in the bracket are still being decided by intercontinental playoffs. These are treated as average-ranked teams in the simulation.

**Elo is stateless between sessions.** The base Elo ratings are fixed at the last historical match (March 2026). Live results entered in the app update Elo for the session but do not persist between visits.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| pandas | Data loading, cleaning, merging |
| NumPy | Numerical operations, simulation |
| scikit-learn | Model training, evaluation, preprocessing |
| XGBoost | Gradient boosting classifier (comparison) |
| LightGBM | Gradient boosting classifier (comparison) |
| Matplotlib / Seaborn | Visualisations |
| Streamlit | Interactive web application |
| pickle | Model serialisation |

---

## What I Learned

Building this project end-to-end required navigating several non-obvious challenges:

**Data alignment.** Match dates and ranking dates are completely different time series. Using a standard merge would have returned no matches. `merge_asof()` is the correct tool — it finds the most recent ranking snapshot before each match date, not an exact match.

**Leakage prevention.** Every feature — Elo ratings, form windows, head-to-head records — had to be computed using only past data at the time of each match. This required looping through matches chronologically rather than applying vectorised operations across the whole dataset.

**Metric choice.** Accuracy is the wrong metric for football prediction. A model that always predicts home win achieves ~48% accuracy without learning anything. Log-loss penalises confident wrong predictions heavily and rewards well-calibrated probabilities — which is what a simulation system actually needs.

**Simulation speed.** The naive approach of calling `model.predict_proba()` for each match inside a 10,000-iteration loop is orders of magnitude too slow. Precomputing all pairwise probabilities in a single batch before the simulation starts reduces runtime from minutes to seconds.

**Model selection.** Logistic Regression outperforming XGBoost and LightGBM on this dataset is a meaningful result — it means the features are informative and the signal is linear enough that a simple model captures it well. Adding model complexity without adding feature quality does not help.

---

## Author

**Iyinoluwa Don-Taiwo**
B.Sc. Computer Science · McPherson University, Ogun State, Nigeria
ML Intern · Interspatial Technologies, Lagos
DSN Campus Ambassador · PyClub McPherson Lead

Instagram: [@donlovesml_](https://www.instagram.com/donlovesml_/)

---

## Acknowledgements

- [martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — international football results dataset
- [cashncarry](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) — FIFA world rankings dataset
- FIFA — confirmed 2026 World Cup group draw

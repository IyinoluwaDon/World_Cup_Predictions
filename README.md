# 🏆 FIFA World Cup 2026 — ML Prediction System

**Predicting match outcomes and simulating the 2026 FIFA World Cup using machine learning.**

Built by [Iyinoluwa Don-Taiwo](https://www.instagram.com/donlovesml_/)

Live Demo -> [Live APP](https://wc2026prediction.streamlit.app/)

---

## Overview

This project builds an end-to-end machine learning system that predicts the outcomes of international football matches and simulates the entire 2026 FIFA World Cup tournament. The system is trained on 11,437 competitive international matches spanning 2002 to 2026, uses weighted Elo ratings computed from scratch alongside FIFA ranking data and Poisson goal modelling, and runs 10,000 Monte Carlo simulations of the 48-team bracket to produce win probabilities for every competing nation.

The Streamlit web app allows fully interactive predictions — including live result entry shared across all devices via Supabase, model retraining on each new result, predicted scorelines powered by Poisson regression, a full head-to-head history tool, and a real-time accuracy tracker showing how the model performs against actual tournament results.

---

## Live Demo

```bash
streamlit run wc2026_app.py
```

---

## Project Structure

```
world_cup_predictions/
│
├── wc2026_app.py               # Streamlit web application 
├── requirements.txt            # Python dependencies
├── README.md                   
│
├── data/
│   ├── raw/
│   │   ├── results.csv                    # International match results (martj42, Kaggle)
│   │   └── fifa_ranking-2024-06-20.csv    # FIFA world rankings (cashncarry, Kaggle)
│   └── processed/
│       └── master_matches.csv             # Merged, cleaned match dataset
│
├── models/
│   └── best_model.pkl          # Trained model + Poisson models + Elo state (required at runtime)
│
├── outputs/
│   ├── eda_distributions.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── win_probabilities.png
│   ├── top10_probabilities.png
│   └── model_comparison.png
│
└── notebook.ipynb              # Full ML pipeline — run end-to-end to reproduce best_model.pkl
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

**Why only two datasets?** A third dataset (FIFA 21 Player Ratings) was evaluated and dropped — it is a static 2020/2021 snapshot, covers club ratings rather than national team context, and is missing players who emerged after 2021. FIFA Rankings serve as a superior, time-aligned proxy for team strength.

**Why competitive matches only?** Friendlies are played with rotated squads and low stakes. The dataset keeps: FIFA World Cup, World Cup qualification, UEFA Euro, UEFA Nations League, Copa América, AFCON, AFC Asian Cup, Gold Cup, CONCACAF Nations League.

**Why 2002 onward?** Football before this era is structurally too different — tactics, fitness, global participation, and data quality all changed substantially. Post-2002 matches are more predictive of 2026 outcomes.

---

## Data Pipeline

### Step 1 — Team name normalisation

The two datasets use inconsistent naming conventions. A name map was applied before merging:

```python
name_map = {
    'Czech Republic': 'Czechia',
    'DR Congo':       'Congo DR',
    'Cape Verde':     'Cabo Verde',
    'Curaçao':        'Curacao',
}
```

### Step 2 — Date-aligned ranking merge

FIFA publishes rankings on a monthly schedule, not daily. `pandas.merge_asof()` finds the most recent ranking published before or on each match date — for both teams simultaneously. A standard merge on exact dates would return no results.

### Step 3 — Missing value handling

Matches after the rankings cutoff (June 2024) were filled with each team's last known ranking using forward-fill.

**Final dataset:** 11,437 rows · 2002–2026

---

## Feature Engineering

20 features were engineered from raw match data. All are computed using only past data — never the current or future match — eliminating data leakage.

| Feature | Description |
|---------|-------------|
| `elo_diff_w` | Weighted Elo gap — K varies by tournament importance (WC=60, qualifier=25) |
| `ranking_diff` | FIFA points gap at match time |
| `home_form_5` / `away_form_5` | Win rate in last 5 matches |
| `form_diff` | Net short-term momentum difference |
| `home_form_10` / `away_form_10` | Win rate in last 10 matches |
| `form_diff_10` | Net medium-term momentum difference |
| `home_goals_scored_5` / `away_goals_scored_5` | Average goals scored in last 5 matches |
| `home_goals_conceded_5` / `away_goals_conceded_5` | Average goals conceded in last 5 matches |
| `goal_diff_scored` | Relative attacking gap |
| `goal_diff_conceded` | Relative defensive gap |
| `h2h_home_wins` / `h2h_draws` / `h2h_home_losses` | Head-to-head history |
| `is_neutral` | Neutral venue flag |
| `same_confederation` | Both teams from same confederation |
| `tournament_importance` | Match stakes: WC=3, major tournament=2, qualifier=1 |

### Weighted Elo ratings

Elo is a dynamic rating system that updates after every match. Every team starts at 1500. The K-factor varies by tournament — a World Cup result moves ratings more than a qualifier result, reflecting real-world importance.

```python
K = get_k(tournament)  # WC=60, major=50, qualifier=25-30
expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
elo[home] = home_elo + K * (actual - expected)
```

---

## Model Building

### Train / Calibration / Test Split

Data was split by time, not randomly. Random splits cause data leakage in time-series data.

| Split | Date Range | Matches |
|-------|-----------|---------|
| Training | Before 2018 | 6,591 |
| Calibration | 2018–2021 | ~2,100 |
| Test | 2022 onward | 2,718 |

### Model Comparison

| Model | Accuracy | Log-Loss |
|-------|----------|----------|
| **Logistic Regression (tuned + calibrated)** | **0.5971** | **0.8775** ✓ |
| LightGBM (calibrated) | 0.5946 | 0.9283 |
| Random Forest (calibrated) | 0.6063 | 0.9092 |
| XGBoost (calibrated) | 0.6015 | 0.9383 |
| Soft-Vote Ensemble | 0.6093 | 0.8782 |

Logistic Regression wins on log-loss — the metric that matters for the simulation. The features are doing the heavy lifting. A simpler model with well-engineered features outperforms complex models with the same features on this type of data because the underlying signal is mostly linear and the noise floor is high.

Models were calibrated using isotonic regression on a held-out 2018–2021 calibration set to correct systematic probability biases.

### Classification Report (Logistic Regression, test set)

```
              precision    recall    f1-score

   Away Win       0.59      0.62        0.61
       Draw       0.25      0.04        0.07
   Home Win       0.62      0.85        0.72

   accuracy                             0.60
```

Draw prediction has an F1 of 0.07. This is a known limitation across all football prediction models — draws depend on in-match dynamics no pre-match model can see.

---

## Poisson Goals Model

Alongside the classification model, two separate Poisson regression models predict expected goals (xG) for each team:

- `poisson_home` → predicts expected home team goals
- `poisson_away` → predicts expected away team goals

Score probabilities are derived by computing P(home scores h) × P(away scores a) across all (h, a) combinations. This powers the **Scoreline / xG** tab in the app, showing the most likely scorelines and a full score probability matrix.

The Logistic Regression model is used for all win/draw/loss predictions (better log-loss). Poisson is used exclusively for scoreline display.

---

## Tournament Simulation

### Approach

The model runs 10,000 full tournament simulations. Each simulation:

1. Simulates all 12 groups using probabilistic match sampling with proper FIFA tiebreakers (H2H points → H2H goal difference → overall GD → goals scored)
2. Advances the top 2 from each group + 8 best third-place teams (32 total) to the Round of 32
3. Runs five knockout rounds to a champion
4. Records the winner

Host nations (USA, Canada, Mexico) receive partial home advantage in simulations.

### Performance optimisation

All 2,256 pairwise matchup probabilities for the 48 teams are precomputed in a single vectorised batch before the simulation loop begins. The simulation uses only dictionary lookups — reducing runtime from minutes to seconds.

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

Random baseline (equal probability): **2.08%**

---

## Streamlit App — Feature Guide

The web app has seven fully interactive tabs.

### ⚽ Match Predictor
Select any two World Cup teams and get live win/draw/loss probabilities. Includes a neutral venue toggle and a side-by-side stat comparison table showing Elo, FIFA points, form, and goals per game.

### 🎯 Scoreline / xG
Powered by the Poisson goals model. Shows expected goals for each team, the six most likely scorelines with probabilities, and a full score probability matrix (heatmap) for all scorelines up to 5-5.

### 🏟️ Knockout Simulator
Choose 16 teams for the Round of 16 and simulate the full bracket. The model predicts probabilities for each match and samples the winner probabilistically — run it multiple times and the bracket changes.

### 📊 Win Probabilities
Run Monte Carlo simulations on demand — 1,000, 2,000, 5,000, or 10,000 runs. Shows the full ranked win probability table with Semifinals reach % and Finals reach % per team, coloured by confederation.

### 📋 Group Stage
Pick any of the 12 groups and simulate all 6 round-robin matches with proper FIFA tiebreakers. Shows predicted standings with points, goal difference, and qualifier status.

### 🔍 Head-to-Head Deep Dive
Full historical record between any two teams drawn from 11,437 competitive matches. Shows win/draw/loss breakdown, match timeline with scores and tournaments, and a goals-per-match chart over time.

### 📡 Live Results & Accuracy
Enter actual World Cup scores. Results save to a shared Supabase database — visible to all users on any device instantly. Each new result retrains the Logistic Regression model on the expanded dataset. The accuracy tracker shows the model's predicted outcome vs the actual result for every match entered, with a running accuracy line chart over the tournament.

---

## Shared Live Results — How It Works

```
You enter a score on your phone
        ↓
Score saves to Supabase (shared database)
        ↓
Anyone opens the site on any device
        ↓
App reads from Supabase → sees the score
        ↓
Model retrains on original data + all live results
        ↓
Elo ratings update → all predictions shift
        ↓
Accuracy tracker records predicted vs actual
```

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

The app opens at `http://localhost:8501`.

To retrain the model from scratch, run all cells in `notebook.ipynb`. This regenerates `models/best_model.pkl` and all charts in `outputs/`.

---

## Deploy to Streamlit Community Cloud

```
1. Push the whole repo (including models/best_model.pkl, requirements.txt, packages.txt) to GitHub
2. Go to share.streamlit.io
3. Sign in with GitHub → New app → select repo → set main file to wc2026_app.py
4. Deploy
```

Free tier. No credit card. Deploys in a couple of minutes.

---

## Limitations

**Draws are hard.** F1 of 0.07 on draw prediction. Draws depend on in-match dynamics — pace of play, substitutions, red cards — that no pre-match model can see. This is a ceiling shared by all football prediction models.

**Rankings cutoff.** FIFA rankings data cuts off at June 2024. Matches from June 2024 to March 2026 use each team's last known ranking via forward-fill.

**Squad changes.** The model does not know about injuries, suspensions, or manager changes at match time.

**Elo is session-based.** The base Elo ratings are fixed at the last historical match (March 2026). Live results entered in the app update Elo for the session and persist in Supabase, but the base model is retrained rather than the Elo state being permanently written to disk.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| pandas | Data loading, cleaning, merging |
| NumPy | Numerical operations, simulation |
| scikit-learn | Model training, calibration, evaluation |
| XGBoost / LightGBM | Gradient boosting comparison models |
| SciPy | Poisson distribution for xG modelling |
| Matplotlib | Visualisations |
| Streamlit | Interactive web application |
| Supabase | Shared live results database |
| pickle | Model serialisation |

---

## What I Learned

**Data alignment.** Match dates and ranking dates are completely different time series. `merge_asof()` is the correct tool — it finds the most recent ranking snapshot before each match date rather than requiring an exact date match.

**Leakage prevention.** Every feature — Elo ratings, form windows, goals, head-to-head records — had to be computed using only past data at the time of each match. This required looping chronologically rather than applying vectorised operations across the whole dataset.

**Metric choice.** Accuracy is the wrong metric for football prediction. Log-loss penalises confident wrong predictions heavily and rewards well-calibrated probabilities — which is what a simulation system actually needs.

**Calibration matters.** Even a well-trained model can produce systematically over- or under-confident probabilities. Isotonic calibration on a held-out validation set corrects these biases without changing the underlying model.

**Simulation speed.** Calling `model.predict_proba()` for each match inside 10,000 loops is too slow. Precomputing all pairwise probabilities in a single batch and using dictionary lookups inside the loop reduces simulation time from minutes to seconds.

**Model selection.** Logistic Regression outperforming XGBoost and LightGBM on this dataset is a meaningful result — the features are well-engineered and the signal is linear enough that a simple model captures it cleanly. Adding model complexity without adding feature quality does not help on high-noise sports data.

**Poisson modelling.** Predicting goals directly rather than match outcomes gives a richer view of match dynamics. Expected goals capture attacking and defensive quality in a way that Win/Draw/Loss probabilities alone do not.

---

## Author

**Iyinoluwa Don-Taiwo**
ML Engineer · Lagos, Nigeria

Instagram: [@donlovesml_](https://www.instagram.com/donlovesml_/)

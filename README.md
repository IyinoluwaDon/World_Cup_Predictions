# International Football Match Prediction and Tournament Simulation

A historical football modeling project that combines match-outcome classification, team-strength features, scoreline modeling, and tournament simulation. The Streamlit application is the interface to the data and saved model artifacts; `scripts/build_model.py` contains the feature engineering and training workflow.

The pipeline sorts international matches chronologically, constructs weighted Elo ratings from prior matches, adds recent team-form features, and compares logistic regression, Random Forest, XGBoost, and LightGBM outcome classifiers. Poisson regression estimates goal rates for scoreline simulations; time-ordered validation and probability calibration are part of the experiment. Model outputs and tournament simulation inputs are stored under `models/`, `data/`, and `outputs/` when present.

## Run the app

```sh
python -m pip install -r requirements.txt
streamlit run wc2026_app.py
```

Build help and required input paths with `python scripts/build_model.py --help`. `notebooks/` records analysis and `frontend/` contains supporting UI assets.

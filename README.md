# International Football Match Prediction Experiment

A Streamlit project for exploring historical international match outcomes, team ratings and tournament simulations. The repository includes research code and may require local data/model artifacts for the complete interactive experience.

## What is included

- Historical match-data feature engineering and model experiments
- Elo-style ratings and scoreline/tournament simulation code
- Main-guarded training entry point with a dependency-light help command

## Run locally

```sh
python -m pip install -r requirements.txt
streamlit run wc2026_app.py
```

## Verify

python scripts/build_model.py --help

## Repository map

`wc2026_app.py` is the Streamlit entry point. `scripts/` contains training/build tools, `notebooks/` records analysis, and `frontend/` holds supporting UI assets. Check required data/model paths before running a full experiment.

## Evidence and limitations

The training entry point's help and source parsing pass. A full retraining, tournament simulation and held-out retrospective evaluation were not run in this maintenance pass.

Predictions are experimental and depend on data date, tournament format and model assumptions. Verify every dataset's provenance and license. Historical results do not establish future match accuracy; evaluate on tournaments excluded from training before presenting performance claims. Remove or update any stale 2026/live-demo claims before public release.

## License

No repository license file is included. Unless the project owner adds a license, do not assume the code, data or model artifacts may be redistributed or reused. Verify third-party data and model terms separately.

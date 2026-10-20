# 🧪 ExperimentLab — A/B Testing & Experimentation Workbench

A deployable Streamlit app for end-to-end experiment analysis, plus a
first-principles statistics library (`src/stats_core.py`) covered by
22 unit tests.

## What it does

| Tab | Method |
|---|---|
| ⚡ Power analysis | Per-arm sample size, achieved-power curve, MDE solver |
| 🚦 SRM check | χ² sample-ratio-mismatch test (unequal/allocation-aware) |
| 📊 Frequentist test | Two-proportion z-test · Welch's t-test · Mann-Whitney U (bootstrap CI) |
| 🎲 Bayesian test | Beta-Binomial posteriors, P(B>A), expected loss of shipping |
| ✂️ CUPED | Pre-period covariate adjustment with variance-reduction readout |

Data: 3 seeded synthetic experiment generators (conversion with injectable
SRM, heavy-tailed revenue, CUPED demo) **or upload your own CSV**.

## Run locally

```bash
cd experimentlab
pip install -r requirements.txt
streamlit run app.py
pytest tests -q
```

## Deploy (Streamlit Community Cloud, free)

1. Push this folder to a public GitHub repo (as its own repo, with `app.py`
   at the repo root, or as a subfolder — point the Cloud dashboard at
   `experimentlab/app.py` either way).
2. Go to <https://share.streamlit.io> → *New app* → select the repo/branch
   and `experimentlab/app.py`.
3. That's it — dependencies come from `requirements.txt`.

To deploy on Hugging Face Spaces instead: create a *Streamlit* Space and
upload the folder; add `sdk: streamlit` front-matter or an equivalent
`README.md` (HF auto-detects `app.py`).

## Files

- `app.py` — Streamlit UI (Plotly charts, cached data loaders)
- `src/stats_core.py` — power/SRM/frequentist/Bayesian/CUPED (no UI imports)
- `src/demo_data.py` — reproducible synthetic experiments
- `tests/test_stats_core.py` — 22 tests, incl. cross-checks against
  R's `power.prop.test` and scipy reference values

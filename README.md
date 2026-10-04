# Predictive OS

**Predict → Gate → Audit**

Predictive OS is a hackathon project that turns industrial sensor telemetry into governed predictive-maintenance decisions. It helps an operations team benchmark candidate ML models, make confidence-aware predictions, and route uncertain cases to human review instead of automating them blindly.

## What it does

- Uploads a training telemetry CSV and prepares it for modelling.
- Benchmarks Random Forest, XGBoost, and LightGBM using cross-validation.
- Selects the strongest validated candidate for classification or regression.
- Scores unseen telemetry and applies a configurable confidence threshold.
- Sends low-confidence decisions to a human-review queue.
- Shows model metrics, feature importance, data exploration, and business-impact indicators.
- Keeps an audit timeline and exports decision CSVs and a governance PDF.

## Why it matters

Predictive models can be useful, but production decisions need controls. Predictive OS combines model selection with a confidence gate and an audit trail so teams can automate high-confidence cases while retaining human oversight where risk is higher.

## Built with

- Python and Streamlit
- pandas and NumPy
- scikit-learn, XGBoost, and LightGBM
- Plotly
- ReportLab
- windsurf(debugging)
- Gemini(Planning)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, then use the following demo flow.

## Demo flow

1. On **Overview**, upload a training telemetry CSV.
2. Select the target column and task type, then train the model benchmark.
3. Review validation results, feature importance, and data exploration.
4. In **Make Predictions**, upload unseen telemetry.
5. Inspect the confidence-gated decisions and download the decision CSV.
6. Visit **Intelligence Reports** for business-impact metrics and the governance PDF export.

## Deployment

The live demo is deployed with Streamlit Community Cloud. In Streamlit Cloud, select this repository and set the main file path to `app.py`.

## Acknowledgements

Gemini was used for planning, and Windsurf was used for debugging during development.

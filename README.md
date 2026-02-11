# AI Insurance Risk Platform (Underwriting + Fraud)

A lightweight decision-support platform that simulates real insurance workflows:
- **Underwriting module:** predicts expected annual medical cost and assigns a risk tier (Low/Medium/High)
- **Claims module:** predicts fraud risk (Y/N) using a supervised ML classifier

Built with **Python, scikit-learn, pandas, and Streamlit**.

## Demo Features
**Underwriting**
- Expected annual cost prediction
- Risk tiering using dataset percentiles
- Risk score (0–100) + progress bar
- Primary risk drivers (rule-based interpretability layer)

**Claims/Fraud**
- Fraud prediction (Y/N)
- Handles categorical feature encoding + column alignment for stable inference

## Data
- Medical cost dataset: `insurance.csv` (Kaggle “Medical Cost Personal Dataset”)
- Claims dataset: `claims.csv` (insurance claims/fraud dataset)

> Note: Datasets are not included in this repo. Place them in `data/` as:
> - `data/insurance.csv`
> - `data/claims.csv`

## How to Run Locally
```bash
pip install pandas scikit-learn streamlit joblib matplotlib
streamlit run app/streamlit_app.pyapp/        # Streamlit app
src/        # training scripts
data/       # local datasets (not tracked)
models/     # trained models (not tracked)

# Lauki Finance: Credit Risk Modelling

A machine learning-powered Streamlit web app that predicts credit risk for loan applicants. The app calculates default probability, credit score, and risk rating based on applicant financial and demographic data.

**Live App:** [https://credit-risk-model-stramlit-git-4yntqsz8hdckaujhrecke7.streamlit.app/](https://credit-risk-model-stramlit-git-4yntqsz8hdckaujhrecke7.streamlit.app/)

## Features

- **Default Probability**: Predicts the likelihood of loan default using a logistic regression model
- **Credit Score**: Generates a credit score on a 300–900 scale
- **Risk Rating**: Classifies applicants into Poor, Average, Good, or Excellent rating tiers

## Input Parameters

| Parameter | Description |
|---|---|
| Age | Applicant age (18–100) |
| Income | Annual income |
| Loan Amount | Requested loan amount |
| Loan Tenure | Loan duration in months |
| Avg DPD | Average days past due per delinquency |
| Delinquency Ratio | Percentage of delinquent accounts |
| Credit Utilization Ratio | Credit utilization percentage |
| Open Loan Accounts | Number of currently open accounts |
| Residence Type | Owned / Rented / Mortgage |
| Loan Purpose | Education / Home / Auto / Personal |
| Loan Type | Secured / Unsecured |

## How It Works

1. User inputs applicant data via the Streamlit UI
2. Inputs are transformed (one-hot encoding, scaling) to match the trained model's feature space
3. A logistic regression model computes the default probability
4. Credit score is derived from the non-default probability, mapped to a 300–900 range
5. A rating label (Poor / Average / Good / Excellent) is assigned based on the score

## Project Structure

```
credit-risk-model-stramlit/
├── app/
│   ├── main.py                  # Streamlit UI
│   └── prediction_helper.py     # ML inference logic
├── artifacts/
│   └── model_data.joblib        # Trained model, scaler, feature config
├── dataset/
│   ├── customers.csv            # Customer data
│   ├── loans.csv                # Loan data
│   └── bureau_data.csv          # Credit bureau data
├── credit_risk_model.ipynb      # Model training notebook
├── requirements.txt             # Python dependencies
└── README.md
```

## Tech Stack

- **Python 3.x**
- **Streamlit** — Web UI framework
- **scikit-learn** — Logistic regression model
- **pandas / numpy** — Data processing
- **joblib** — Model serialization

## Running Locally

```bash
# Clone the repository
git clone <repo-url>
cd credit-risk-model-stramlit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

The app will open at `http://localhost:8501`.

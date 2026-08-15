# Predictive Modeling and Risk Scoring for Bank Customer Churn

A Streamlit application for predicting bank customer churn probability and supporting retention prioritization.

## Dataset
10,000 customer records from `European_Bank.csv`.

## Final model
Gradient Boosting

## Features
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- BalanceSalaryRatio
- ProductDensity
- EngagementProduct
- AgeTenureInteraction

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Deployment
Push this project folder to GitHub and deploy `app/streamlit_app.py` using Streamlit Community Cloud.

## Important
The churn probability is a predictive risk score, not a causal statement. Risk thresholds should be calibrated using actual retention costs and outcomes before operational use.

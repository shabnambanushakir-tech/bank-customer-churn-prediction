
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
RESULTS_PATH = BASE_DIR / "model_results.csv"
METADATA_PATH = BASE_DIR / "metadata.json"

st.set_page_config(
    page_title="Bank Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)

@st.cache_data
def load_metadata():
    return json.loads(METADATA_PATH.read_text())

def add_features(frame):
    frame = frame.copy()
    frame["BalanceSalaryRatio"] = frame["Balance"] / (frame["EstimatedSalary"] + 1.0)
    frame["ProductDensity"] = frame["NumOfProducts"] / (frame["Tenure"] + 1.0)
    frame["EngagementProduct"] = frame["IsActiveMember"] * frame["NumOfProducts"]
    frame["AgeTenureInteraction"] = frame["Age"] * frame["Tenure"]
    return frame

def risk_category(prob):
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    return "High"

def risk_action(risk):
    if risk == "High":
        return "Prioritize for retention review and personalized engagement."
    if risk == "Medium":
        return "Monitor engagement and consider a targeted retention offer."
    return "Maintain normal engagement and monitor for future risk changes."

model = load_model()
results = load_results()
metadata = load_metadata()

st.title("🏦 Bank Customer Churn Intelligence")
st.caption(
    "Predictive modeling and risk scoring for proactive customer-retention analysis"
)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Risk Calculator", "What-If Simulator", "Model Performance"]
)

# Dashboard
if page == "Dashboard":
    st.header("Executive Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers analyzed", "10,000")
    c2.metric("Observed churn rate", f"{metadata['overall_churn_rate']:.1%}")
    c3.metric("Final model", metadata["model_name"])
    c4.metric("Risk scoring", "Probability-based")

    st.divider()

    st.subheader("Project objective")
    st.write(
        "This application estimates the probability that a bank customer will churn. "
        "The score is intended to support prioritization of retention reviews; it is "
        "not a causal claim that any individual feature causes churn."
    )

    st.subheader("Key EDA insights")
    insight_cols = st.columns(3)
    with insight_cols[0]:
        st.markdown("**Germany**")
        st.write("Observed churn rate: 32.44%, substantially above France and Spain.")
    with insight_cols[1]:
        st.markdown("**Engagement**")
        st.write("Inactive customers show a 26.85% observed churn rate versus 14.27% for active customers.")
    with insight_cols[2]:
        st.markdown("**Age**")
        st.write("Churned customers average 44.84 years versus 37.41 among retained customers.")

    st.subheader("Risk interpretation")
    st.info(
        "Low: <30% probability | Medium: 30–60% | High: ≥60%. "
        "These thresholds are project defaults and should be calibrated against business intervention costs."
    )

# Risk calculator
elif page == "Risk Calculator":
    st.header("Customer Churn Risk Calculator")

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.number_input("Credit Score", 300, 900, 650, 1)
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 18, 100, 40, 1)

        with col2:
            tenure = st.number_input("Tenure (years)", 0, 20, 5, 1)
            balance = st.number_input("Balance", 0.0, 500000.0, 75000.0, 1000.0)
            products = st.number_input("Number of Products", 1, 4, 2, 1)
            has_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col3:
            active = st.selectbox("Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No")
            salary = st.number_input("Estimated Salary", 0.0, 500000.0, 100000.0, 1000.0)

        submitted = st.form_submit_button("Calculate Churn Risk", type="primary")

    if submitted:
        customer = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": active,
            "EstimatedSalary": salary
        }])

        customer = add_features(customer)
        probability = float(model.predict_proba(customer)[0, 1])
        risk = risk_category(probability)

        st.divider()
        a, b, c = st.columns(3)
        a.metric("Churn probability", f"{probability:.1%}")
        b.metric("Risk category", risk)
        c.metric("Binary churn flag", "Likely churn" if probability >= 0.50 else "Likely retained")

        if risk == "High":
            st.error(f"🔴 HIGH RISK — {risk_action(risk)}")
        elif risk == "Medium":
            st.warning(f"🟠 MEDIUM RISK — {risk_action(risk)}")
        else:
            st.success(f"🟢 LOW RISK — {risk_action(risk)}")

        st.subheader("Engineered model inputs")
        display_features = customer[
            ["BalanceSalaryRatio", "ProductDensity",
             "EngagementProduct", "AgeTenureInteraction"]
        ].T.rename(columns={0: "Value"})
        st.dataframe(display_features, use_container_width=True)

# What-if simulator
elif page == "What-If Simulator":
    st.header("What-If Churn Risk Simulator")
    st.write(
        "Change customer characteristics and compare the model's predicted probability. "
        "This is a hypothetical model scenario, not a guaranteed causal effect."
    )

    defaults = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 5,
        "Balance": 75000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 100000.0
    }

    left, right = st.columns(2)

    with left:
        st.subheader("Current customer")
        current = defaults.copy()
        current["CreditScore"] = st.number_input("Current credit score", 300, 900, 650, key="cur_cs")
        current["Age"] = st.number_input("Current age", 18, 100, 40, key="cur_age")
        current["Balance"] = st.number_input("Current balance", 0.0, 500000.0, 75000.0, 1000.0, key="cur_bal")
        current["NumOfProducts"] = st.number_input("Current products", 1, 4, 2, key="cur_prod")
        current["IsActiveMember"] = st.selectbox("Current active member", [0, 1], index=1, key="cur_active")
        current["Geography"] = st.selectbox("Current geography", ["France", "Spain", "Germany"], key="cur_geo")
        current["Gender"] = st.selectbox("Current gender", ["Male", "Female"], key="cur_gender")
        current["Tenure"] = st.number_input("Current tenure", 0, 20, 5, key="cur_tenure")
        current["HasCrCard"] = st.selectbox("Current credit card", [0, 1], index=1, key="cur_card")
        current["EstimatedSalary"] = st.number_input("Current salary", 0.0, 500000.0, 100000.0, 1000.0, key="cur_salary")

    with right:
        st.subheader("Scenario")
        scenario = current.copy()
        scenario["NumOfProducts"] = st.number_input("Scenario products", 1, 4, current["NumOfProducts"], key="sc_prod")
        scenario["IsActiveMember"] = st.selectbox("Scenario active member", [0, 1], index=current["IsActiveMember"], key="sc_active")
        scenario["Balance"] = st.number_input("Scenario balance", 0.0, 500000.0, current["Balance"], 1000.0, key="sc_bal")
        scenario["Age"] = st.number_input("Scenario age", 18, 100, current["Age"], key="sc_age")

    current_df = add_features(pd.DataFrame([current]))
    scenario_df = add_features(pd.DataFrame([scenario]))

    current_prob = float(model.predict_proba(current_df)[0, 1])
    scenario_prob = float(model.predict_proba(scenario_df)[0, 1])
    change = scenario_prob - current_prob

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Current probability", f"{current_prob:.1%}")
    b.metric("Scenario probability", f"{scenario_prob:.1%}")
    c.metric("Change", f"{change:+.1%}")

    if change < 0:
        st.success("The scenario lowers the model's predicted churn probability.")
    elif change > 0:
        st.warning("The scenario increases the model's predicted churn probability.")
    else:
        st.info("The scenario produces the same predicted probability.")

# Model performance
elif page == "Model Performance":
    st.header("Model Performance")

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        use_container_width=True
    )

    st.subheader("ROC-AUC comparison")
    chart = results.set_index("Model")["ROC-AUC"].sort_values()
    st.bar_chart(chart)

    st.subheader("Interpretation")
    st.write(
        "ROC-AUC measures how well the model separates churners from retained customers "
        "across classification thresholds. Precision measures the reliability of churn "
        "alerts, while recall measures how many actual churners are captured."
    )

st.sidebar.divider()
st.sidebar.caption("Bank Customer Churn Intelligence • ML project")

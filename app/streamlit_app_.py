import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "European_Bank.csv"
RESULTS_PATH = BASE_DIR / "model_results.csv"
METADATA_PATH = BASE_DIR / "metadata.json"

st.set_page_config(
    page_title="Bank Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(DATA_PATH, encoding="latin1")

def add_features(frame):
    frame = frame.copy()
    frame["BalanceSalaryRatio"] = frame["Balance"] / (frame["EstimatedSalary"] + 1.0)
    frame["ProductDensity"] = frame["NumOfProducts"] / (frame["Tenure"] + 1.0)
    frame["EngagementProduct"] = frame["IsActiveMember"] * frame["NumOfProducts"]
    frame["AgeTenureInteraction"] = frame["Age"] * frame["Tenure"]
    return frame

@st.cache_resource
def train_model():
    df = load_data().copy()

    X = df.drop(columns=["Exited", "CustomerId", "Surname"], errors="ignore")
    y = df["Exited"].astype(int)

    X = add_features(X)

    categorical = ["Geography", "Gender"]
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }

    return pipeline, metrics

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

model, live_metrics = train_model()
df = load_data()

try:
    metadata = json.loads(METADATA_PATH.read_text())
except Exception:
    metadata = {
        "overall_churn_rate": float(df["Exited"].mean()),
        "model_name": "Gradient Boosting"
    }

st.title("🏦 Bank Customer Churn Intelligence")
st.caption("Predictive modeling and risk scoring for proactive customer-retention analysis")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Risk Calculator", "What-If Simulator", "Model Performance"]
)

if page == "Dashboard":
    st.header("Executive Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers analyzed", f"{len(df):,}")
    c2.metric("Observed churn rate", f"{df['Exited'].mean():.1%}")
    c3.metric("Final model", "Gradient Boosting")
    c4.metric("ROC-AUC", f"{live_metrics['ROC-AUC']:.3f}")

    st.divider()
    st.subheader("Project objective")
    st.write(
        "This application estimates the probability that a bank customer will churn. "
        "The score supports prioritization of retention reviews; it is not a causal claim."
    )

    st.subheader("Key EDA insights")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Germany**")
        st.write("Observed churn rate: 32.44%, substantially above France and Spain.")
    with cols[1]:
        st.markdown("**Engagement**")
        st.write("Inactive customers show higher observed churn than active customers.")
    with cols[2]:
        st.markdown("**Age**")
        st.write("Churned customers are older on average than retained customers.")

    st.subheader("Churn distribution")
    st.bar_chart(df["Exited"].value_counts().rename({0: "Retained", 1: "Churned"}))

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

elif page == "What-If Simulator":
    st.header("What-If Churn Risk Simulator")
    st.write("Compare the model's predicted probability before and after hypothetical changes.")

    defaults = {
        "CreditScore": 650, "Geography": "France", "Gender": "Male",
        "Age": 40, "Tenure": 5, "Balance": 75000.0,
        "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1,
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

    current_prob = float(model.predict_proba(add_features(pd.DataFrame([current])))[0, 1])
    scenario_prob = float(model.predict_proba(add_features(pd.DataFrame([scenario])))[0, 1])
    change = scenario_prob - current_prob

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

elif page == "Model Performance":
    st.header("Model Performance")
    perf = pd.DataFrame([{"Model": "Gradient Boosting", **live_metrics}])
    st.dataframe(
        perf.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        use_container_width=True
    )
    st.subheader("Performance metrics")
    st.bar_chart(perf.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].T)

st.sidebar.divider()
st.sidebar.caption("Bank Customer Churn Intelligence • ML project")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="Financial Intelligence Dashboard",
    layout="wide"
)

# -----------------------------
# LOAD MODELS & DATA
# -----------------------------
clf = joblib.load("models/cls_model.pkl")
reg = joblib.load("models/reg_model.pkl")
ohe = joblib.load("models/ohe_financial_goal.pkl")

X_train = pd.read_csv("models/X_train.csv")
model_columns = X_train.columns.tolist()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.title("🧾 User Financial Profile")

age = st.sidebar.number_input("Age", 18, 80, 25)
income = st.sidebar.number_input("Monthly Income (₹)", 1000, 200000, 30000)
savings = st.sidebar.number_input("Savings (₹)", 0, 1000000, 5000)
spending = st.sidebar.slider("Spending %", 0, 100, 40)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
expense_ratio = st.sidebar.slider("Expense Ratio %", 0, 100, 40)

budget_score = st.sidebar.slider("Budget Score", 0, 10, 5)
investment_score = st.sidebar.slider("Investment Knowledge", 0, 10, 5)
loan_knowledge_score = st.sidebar.slider("Loan Knowledge", 0, 10, 5)
risk_score = st.sidebar.slider("Risk Appetite", 0, 10, 5)
impulse_purchases = st.sidebar.slider("Impulse Purchases", 0, 10, 3)

goal = st.sidebar.selectbox(
    "Financial Goal",
    ["Debt Repayment", "Emergency Fund", "Investing", "Short Term Savings"]
)

# -----------------------------
# BUILD INPUT DATA
# -----------------------------
goal_encoded = ohe.transform([[goal]])
goal_cols = ohe.get_feature_names_out(["financial_goal"])
goal_df = pd.DataFrame(goal_encoded, columns=goal_cols)

user_numeric = pd.DataFrame([{
    "age": age,
    "income": income,
    "savings": savings,
    "spending": spending,
    "credit_score": credit_score,
    "expense_ratio": expense_ratio,
    "budget_score": budget_score,
    "investment_score": investment_score,
    "loan_knowledge_score": loan_knowledge_score,
    "risk_score": risk_score,
    "impulse_purchases": impulse_purchases,
}])

user_data = pd.concat([user_numeric, goal_df], axis=1)
user_data = user_data.reindex(columns=model_columns, fill_value=0)

# -----------------------------
# DASHBOARD HEADER
# -----------------------------
st.title(" Financial Intelligence Auto Dashboard")
st.caption("ML-powered Financial Literacy & SIP Recommendation System")

# -----------------------------
# PREDICTIONS
# -----------------------------
literacy_pred = clf.predict(user_data)[0]
sip_pred = int(max(reg.predict(user_data)[0], 0))

literacy_map = {0: "Low", 1: "Medium", 2: "High"}

c1, c2, c3 = st.columns(3)

c1.metric(" Literacy Level", literacy_map[literacy_pred])
c2.metric(" Recommended SIP", f"₹ {sip_pred}")
c3.metric(" Financial Goal", goal)

# -----------------------------
# INPUT SUMMARY
# -----------------------------
st.subheader("User Input Summary")
st.dataframe(user_numeric)

# -----------------------------
# SHAP EXPLAINABILITY
# -----------------------------
st.subheader(" Model Explainability (SHAP)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Classification Model")
    try:
        img = Image.open("models/shap_classification.png")
        st.image(img, use_container_width=True)
    except:
        st.warning("Classification SHAP image not found")

with col2:
    st.markdown("### Regression Model")
    try:
        img = Image.open("models/shap_regression.png")
        st.image(img, use_container_width=True)
    except:
        st.warning("Regression SHAP image not found")

# -----------------------------
# MODEL INFO
# -----------------------------
st.subheader(" Model Information")

st.markdown("""
- **Classification Model:** Financial Literacy Level  
- **Regression Model:** SIP Recommendation  
- **Tech Stack:** Python, Scikit-Learn, SHAP, Streamlit  
- **Features:** Financial behavior + goals + discipline scores  
""")

# -----------------------------
# FOOTER
# -----------------------------
st.write("---")


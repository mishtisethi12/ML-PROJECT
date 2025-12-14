# ML-PROJECT
OVERVIEW:

This project is an end-to-end data analytics and machine learning system designed to analyze financial behavior and provide actionable insights. It predicts a user’s financial literacy level and recommends a suitable monthly SIP (Systematic Investment Plan) amount based on income, savings, spending habits, and financial goals.
The project focuses on combining data analysis, machine learning, and explainability to build a transparent and user-centric financial intelligence solution.

PROBLEM STATEMENT:

Many individuals struggle to make informed financial decisions due to limited understanding of budgeting, savings, and investments. Traditional financial advice often lacks personalization and transparency.

This project aims to:

1.Assess financial literacy using data-driven techniques

2.Recommend a sustainable monthly investment amount

3.Provide explainable predictions to build user trust

KEY FEATURES

1.Data cleaning and preprocessing of structured financial data

2.Exploratory Data Analysis (EDA) to understand financial behavior patterns

3.Classification model to predict financial literacy levels (Low / Medium / High)

4.Regression model to estimate a suitable SIP amount

5.Feature engineering on behavioral and financial indicators

6.Model explainability using SHAP

7.Interactive dashboard built with Streamlit

DATASET

The project uses synthetic financial data generated to simulate real-world financial behavior.
Each record represents an individual and includes:

1.Income, savings, and spending behavior

2.Credit score and financial discipline indicators

3.Investment knowledge and risk appetite

4.Financial goals

Synthetic data was chosen to avoid privacy concerns while maintaining realistic patterns.

MACHINE LEARNING APPROACH

1.Classification Model: Predicts financial literacy level

2.Regression Model: Estimates recommended monthly SIP amount

Both models are trained using supervised learning techniques. Feature importance and prediction drivers are analyzed using SHAP to ensure interpretability.

EXPLAINABILTY

SHAP (SHapley Additive exPlanations) is used to:

1.Identify key features influencing predictions

2.Explain model behavior in a transparent manner

3.Improve trust in financial recommendations

Finance project/
├── data/
│   └── synthetic_fin_lit.csv
├── src/
│   ├── preprocess.py
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── shap_explain.py
│   └── streamlit_dashboard.py
├── models/
│   ├── cls_model.pkl
│   ├── reg_model.pkl
│   ├── shap_classification.png
│   ├── shap_regression.png
│   └── X_train.csv
└── README.md

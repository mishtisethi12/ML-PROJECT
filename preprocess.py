# src/01_preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

# Ensure models folder exists (saves outputs here)
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)

# 1) Load raw data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_fin_lit.csv')
df = pd.read_csv(data_path)

# 2) Recompute / ensure derived features
df['save_rate'] = (df['savings'] / (df['income'] + 1)).round(4)
df['debt_to_income'] = (df['debt'] / (df['income'] + 1)).round(4)
df['expense_ratio'] = (df['expenses'] / (df['income'] + 1)).round(4)

# 3) Choose features and targets
features = [
    'age', 'income', 'savings', 'credit_score', 'expense_ratio',
    'budget_score', 'investment_score', 'loan_knowledge_score', 'risk_score',
    'impulse_purchases'
]
cat_feature = ['financial_goal']  # will one-hot encode this

target_cls = 'literacy_level'          # classification target (0/1/2)
target_reg = 'recommended_monthly'    # regression target (SIP amount)

# 4) One-hot encode financial_goal and append to features
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(df[cat_feature])
ohe_cols = list(ohe.get_feature_names_out(cat_feature))
ohe_df = pd.DataFrame(ohe.transform(df[cat_feature]), columns=ohe_cols, index=df.index)

# 5) Build final X (features) dataframe and targets
X = pd.concat([df[features].reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
y_cls = df[target_cls]
y_reg = df[target_reg]

# 6) Train-test split (use stratify for classification target)
X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.20, random_state=42, stratify=y_cls
)

# 7) Save processed files and encoder into models/
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
X_train.to_csv(os.path.join(models_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(models_dir, 'X_test.csv'), index=False)
y_cls_train.to_csv(os.path.join(models_dir, 'y_cls_train.csv'), index=False)
y_cls_test.to_csv(os.path.join(models_dir, 'y_cls_test.csv'), index=False)
y_reg_train.to_csv(os.path.join(models_dir, 'y_reg_train.csv'), index=False)
y_reg_test.to_csv(os.path.join(models_dir, 'y_reg_test.csv'), index=False)

joblib.dump(ohe, os.path.join(models_dir, 'ohe_financial_goal.pkl'))

print("Preprocessing complete — saved X_train/X_test, y_cls/y_reg splits and encoder to models/")

import shap 
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
X_train = pd.read_csv("models/X_train.csv")

# Load models
clf = joblib.load("models/cls_model.pkl")
reg = joblib.load("models/reg_model.pkl")

# Use same sample for SHAP calculation and plotting
sample = X_train[:200]

# ==========================
# Classification SHAP
# ==========================
print("Generating SHAP values for classification model...")
explainer_clf = shap.Explainer(clf, sample)
shap_values_clf = explainer_clf(sample)

plt.title("Classification Model - SHAP Summary")
shap.summary_plot(shap_values_clf.values, sample, show=False)
plt.savefig("models/shap_classification.png")
plt.close()

# ==========================
# Regression SHAP
# ==========================
print("Generating SHAP values for regression model...")
explainer_reg = shap.Explainer(reg, sample)
shap_values_reg = explainer_reg(sample)

plt.title("Regression Model - SHAP Summary")
shap.summary_plot(shap_values_reg.values, sample, show=False)
plt.savefig("models/shap_regression.png")
plt.close()

print("SHAP plots saved:")
print("→ models/shap_classification.png")
print("→ models/shap_regression.png")
# ---------------------------------------------
#  FEATURE IMPORTANCE BAR PLOTS
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Classification Feature Importance
try:
    importances_clf = clf.feature_importances_
    features = X_train.columns

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances_clf)
    plt.title("Classification Model - Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance_classification.png")
    plt.close()
    print("→ Feature importance saved: models/feature_importance_classification.png")

except:
    print("Classification model has no feature_importances_ attribute")

# Regression Feature Importance
try:
    importances_reg = reg.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances_reg)
    plt.title("Regression Model - Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance_regression.png")
    plt.close()
    print("→ Feature importance saved: models/feature_importance_regression.png")

except:
    print("Regression model has no feature_importances_ attribute")

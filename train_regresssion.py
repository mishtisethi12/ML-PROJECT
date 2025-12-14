import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load preprocessed data
X_train = pd.read_csv("models/X_train.csv")
X_test = pd.read_csv("models/X_test.csv")

y_train_reg = pd.read_csv("models/y_reg_train.csv").values.ravel()
y_test_reg = pd.read_csv("models/y_reg_test.csv").values.ravel()

# Train XGBoost Regression model
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)

print("Training regression model...")
model.fit(X_train, y_train_reg)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test_reg, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))

print("\nRegression Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# Save the regression model
joblib.dump(model, "models/reg_model.pkl")

print("\nSaved model → models/reg_model.pkl")

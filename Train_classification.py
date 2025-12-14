import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed CSV files
X_train = pd.read_csv("models/X_train.csv")
X_test = pd.read_csv("models/X_test.csv")
y_train_cls = pd.read_csv("models/y_cls_train.csv").values.ravel()
y_test_cls = pd.read_csv("models/y_cls_test.csv").values.ravel()

# Train XGBoost classification model
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)

print("Training classification model...")
model.fit(X_train, y_train_cls)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred))
print("Accuracy:", accuracy_score(y_test_cls, y_pred))

# Save model
joblib.dump(model, "models/cls_model.pkl")
print("\nModel saved as models/cls_model.pkl")

# Train an XGBoost/LightGBM model for NCAA tournament game prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
import shap
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
FEATURES_CSV = os.path.join(DATA_DIR, 'model_features.csv')
MODEL_OUT = os.path.join(DATA_DIR, 'xgb_model.json')
CALIBRATED_MODEL_OUT = os.path.join(DATA_DIR, 'xgb_model_calibrated.pkl')
SHAP_OUT = os.path.join(DATA_DIR, 'shap_feature_importance.csv')

# Load data
df = pd.read_csv(FEATURES_CSV)

# Minimal features for now (expand as more data is available)
feature_cols = ['favorite_probability', 'round_num', 'year']
X = df[feature_cols]
y = df['favorite_win_flag']

# Robust split: hold out the latest available year as test set
all_years = sorted(df['year'].dropna().unique())
if len(all_years) > 1:
    test_year = all_years[-1]
    train_idx = df['year'] != test_year
    test_idx = df['year'] == test_year
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
else:
    # Fallback: random split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Calibration (Platt scaling)
calibrator = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrator.fit(X_train, y_train)

# Predictions and evaluation
probs = calibrator.predict_proba(X_test)[:, 1]
preds = calibrator.predict(X_test)
acc = accuracy_score(y_test, preds)
ll = log_loss(y_test, probs)
brier = brier_score_loss(y_test, probs)

print(f"Test Accuracy: {acc:.3f}")
print(f"Test Log Loss: {ll:.3f}")
print(f"Test Brier Score: {brier:.3f}")

# Save model
model.save_model(MODEL_OUT)
import joblib
joblib.dump(calibrator, CALIBRATED_MODEL_OUT)

# SHAP explanations
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)
shap_df = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
})
shap_df.to_csv(SHAP_OUT, index=False)
print(f"SHAP feature importance saved to {SHAP_OUT}")

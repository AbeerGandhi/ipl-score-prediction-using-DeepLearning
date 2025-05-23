import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Load data
ipl = pd.read_csv("ipl_data.csv")
ipl.drop(columns=['date', 'mid'], inplace=True, errors='ignore')

# Encode categorical features
cat_features = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
for col in cat_features:
    le = LabelEncoder()
    ipl[col] = le.fit_transform(ipl[col].astype(str))

# Feature Engineering
ipl['run_rate'] = ipl['runs'] / (ipl['overs'] + 1e-5)
ipl['wickets_remaining'] = 10 - ipl['wickets']
ipl['momentum'] = ipl['runs_last_5'] / (ipl['overs'] - 5 + 1e-5)

# Features and Target
features = cat_features + [
    'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
    'striker', 'non-striker',
    'run_rate', 'wickets_remaining', 'momentum'
]
X = ipl[features]
y = ipl['total']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train XGBoost
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" MAE: {mae:.2f}")
print(f" MSE: {mse:.2f}")
print(f" R² Score: {r2:.4f}")


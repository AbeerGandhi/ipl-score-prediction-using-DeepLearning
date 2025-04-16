
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings("ignore")

ipl = pd.read_csv("ipl_data.csv")  # Update the path accordingly
ipl.drop(columns=['date', 'mid'], inplace=True, errors='ignore')  # Drop unnecessary columns

"""# Step 2 : EDA"""

# Display basic dataset info
print("Dataset Overview:")
print(ipl.info())

# Show first & last 5 rows
print("\n🔹 First 5 rows:")
print(ipl.head())

print("\n🔹 Last 5 rows:")
print(ipl.tail())

# Check missing values
print("\n🔹 Missing Values:")
print(ipl.isnull().sum())

# Describe numerical columns
print("\n🔹 Numerical Feature Summary:")
print(ipl.describe())

# Describe categorical columns
print("\n🔹 Categorical Feature Summary:")
print(ipl.describe(include=['object']))

"""Graphs

# Step 3:  Feature Engineering & Preprocessing
"""

#  Step 2: Feature Engineering
cat_features = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
num_features = ['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'striker', 'non-striker']

# Apply Label Encoding
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    ipl[col] = le.fit_transform(ipl[col].astype(str))
    label_encoders[col] = le

# Normalize Numerical Features
scaler = MinMaxScaler()
ipl[num_features] = scaler.fit_transform(ipl[num_features])

#  Prepare Data
X = ipl[cat_features + num_features]
y = ipl['total']  # Target variable

"""# Step 4: Train-Test Split"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# Step 5:  Model Definition & Training"""

# Define Optimized Neural Network
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),

    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate Model
y_pred = model.predict(X_test)

# Evaluate Model
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print(f"MAE: {mean_absolute_error(y_test, y_pred)}, MSE: {mean_squared_error(y_test, y_pred)}, R²: {r2_score(y_test, y_pred)}")

"""# Save the model"""

# Save the trained model
model.save("cricket_score_prediction.h5")
print("Model saved successfully! ")

"""# Step 7:Load the saved model & Prediction UI"""

from tensorflow.keras.models import load_model
import tensorflow as tf
from warnings import filterwarnings
filterwarnings("ignore")
# Load the trained model with explicit loss function
model = load_model("cricket_score_prediction.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

print("Model loaded successfully! ")


# Step 8: Interactive Score Prediction Widget
venue = widgets.Dropdown(options=list(label_encoders['venue'].classes_), description='Venue:')
batting_team = widgets.Dropdown(options=list(label_encoders['bat_team'].classes_), description='Batting Team:')
bowling_team = widgets.Dropdown(options=list(label_encoders['bowl_team'].classes_), description='Bowling Team:')
striker = widgets.Dropdown(options=list(label_encoders['batsman'].classes_), description='Striker:')
bowler = widgets.Dropdown(options=list(label_encoders['bowler'].classes_), description='Bowler:')

runs = widgets.IntText(description='Runs:')
wickets = widgets.IntText(description='Wickets:')
overs = widgets.FloatText(description='Overs:')
runs_last_5 = widgets.IntText(description='Runs Last 5:')
wickets_last_5 = widgets.IntText(description='Wickets Last 5:')
striker_score = widgets.IntText(description='Striker Score:')
non_striker_score = widgets.IntText(description='Non-Striker Score:')

predict_button = widgets.Button(description="Predict Runs")
output = widgets.Output()

def predict_runs(b):
    with output:
        clear_output()

        # Convert categorical inputs
        encoded_input = [
            label_encoders['venue'].transform([venue.value])[0],
            label_encoders['bat_team'].transform([batting_team.value])[0],
            label_encoders['bowl_team'].transform([bowling_team.value])[0],
            label_encoders['batsman'].transform([striker.value])[0],
            label_encoders['bowler'].transform([bowler.value])[0]
        ]

        # Numerical inputs
        numerical_input = [
            runs.value, wickets.value, overs.value,
            runs_last_5.value, wickets_last_5.value,
            striker_score.value, non_striker_score.value
        ]

        # Combine categorical & numerical inputs
        input_data = np.array(encoded_input + numerical_input).reshape(1, -1)

        # Normalize numerical features (last 7 columns)
        input_data[:, -7:] = scaler.transform(input_data[:, -7:])

        # Predict Runs
        predicted_runs = model.predict(input_data)
        print(f"🏏 Predicted Runs: {int(predicted_runs[0,0])}")

# Bind button
predict_button.on_click(predict_runs)

# Display UI
display(venue, batting_team, bowling_team, striker, bowler,
        runs, wickets, overs, runs_last_5, wickets_last_5, striker_score, non_striker_score,
        predict_button, output)

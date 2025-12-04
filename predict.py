import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("sleep_data.csv")

# Create separate encoders
stress_encoder = LabelEncoder()
exercise_encoder = LabelEncoder()
quality_encoder = LabelEncoder()

# Fit encoders on each text column
data['Stress_Level_enc'] = stress_encoder.fit_transform(data['Stress_Level'])
data['Exercise_Frequency_enc'] = exercise_encoder.fit_transform(data['Exercise_Frequency'])
data['Sleep_Quality_enc'] = quality_encoder.fit_transform(data['Sleep_Quality'])

# Features (X) and target (y)
X = data[['Sleep_Duration', 'Stress_Level_enc', 'Exercise_Frequency_enc', 'Coffee_Intake', 'Screen_Time']]
y = data['Sleep_Quality_enc']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# ========= USER INPUT =========

sleep = float(input("Sleep Duration (hours): "))

stress_in = input("Stress Level (Low/Medium/High): ")
exercise_in = input("Exercise Frequency (None/Weekly/Daily): ")
coffee = int(input("Daily Coffee Intake (cups): "))
screen = float(input("Screen Time (hours): "))

# Clean user text (ignore case, spaces)
stress_in_clean = stress_in.strip().lower()
exercise_in_clean = exercise_in.strip().lower()

# Map to correct format used in dataset
stress_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
exercise_map = {'none': 'None', 'weekly': 'Weekly', 'daily': 'Daily'}

if stress_in_clean not in stress_map:
    raise ValueError("Invalid Stress Level. Please use Low/Medium/High.")

if exercise_in_clean not in exercise_map:
    raise ValueError("Invalid Exercise Frequency. Please use None/Weekly/Daily.")

stress_label = stress_map[stress_in_clean]
exercise_label = exercise_map[exercise_in_clean]

# Encode using the SAME encoders as training
stress_encoded = stress_encoder.transform([stress_label])[0]
exercise_encoded = exercise_encoder.transform([exercise_label])[0]

# Create input row
user_data = [[sleep, stress_encoded, exercise_encoded, coffee, screen]]

# Predict
predicted_class = model.predict(user_data)[0]
predicted_label = quality_encoder.inverse_transform([predicted_class])[0]

print("Predicted Sleep Quality:", predicted_label)

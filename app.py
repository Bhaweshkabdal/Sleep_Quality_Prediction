import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("sleep_data.csv")

# Separate encoders
stress_encoder = LabelEncoder()
exercise_encoder = LabelEncoder()
quality_encoder = LabelEncoder()

data['Stress_Level_enc'] = stress_encoder.fit_transform(data['Stress_Level'])
data['Exercise_Frequency_enc'] = exercise_encoder.fit_transform(data['Exercise_Frequency'])
data['Sleep_Quality_enc'] = quality_encoder.fit_transform(data['Sleep_Quality'])

# Training
X = data[['Sleep_Duration', 'Stress_Level_enc', 'Exercise_Frequency_enc', 'Coffee_Intake', 'Screen_Time']]
y = data['Sleep_Quality_enc']
model = RandomForestClassifier()
model.fit(X, y)

# ---------------- Web UI ---------------- #

st.title("Sleep Quality Prediction System")
st.subheader("Enter Your Lifestyle & Sleep Details")

sleep = st.number_input("Sleep Duration (hours)", min_value=1.0, max_value=12.0, value=7.0)

stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
exercise = st.selectbox("Exercise Frequency", ["None", "Weekly", "Daily"])

coffee = st.number_input("Daily Coffee Intake (cups)", min_value=0, max_value=10, value=1)
screen = st.number_input("Screen Time (hours)", min_value=0.0, max_value=12.0, value=4.0)

if st.button("Predict Sleep Quality"):
    stress_enc = stress_encoder.transform([stress])[0]
    exercise_enc = exercise_encoder.transform([exercise])[0]

    user_data = [[sleep, stress_enc, exercise_enc, coffee, screen]]
    pred = model.predict(user_data)[0]

    result = quality_encoder.inverse_transform([pred])[0]

    st.success(f"Predicted Sleep Quality: {result}")

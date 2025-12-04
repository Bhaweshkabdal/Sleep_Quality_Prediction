import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("sleep_data.csv")

# ================== MANUAL ENCODING (No Error Ever) ==================

stress_map = {"Low":0, "Medium":1, "High":2}
exercise_map = {"None":0, "Weekly":1, "Daily":2}
quality_map = {"Poor":0, "Average":1, "Good":2}

# Convert dataset values to numbers
data["Stress_Level"] = data["Stress_Level"].map(stress_map)
data["Exercise_Frequency"] = data["Exercise_Frequency"].map(exercise_map)
data["Sleep_Quality"] = data["Sleep_Quality"].map(quality_map)

# Train Model
X = data[["Sleep_Duration","Stress_Level","Exercise_Frequency","Coffee_Intake","Screen_Time"]]
y = data["Sleep_Quality"]

model = RandomForestClassifier()
model.fit(X,y)

# ==================== STREAMLIT UI ======================

st.title("Sleep Quality Prediction App")
st.write("Enter your daily habits to predict sleep performance")

sleep = st.number_input("Sleep Duration (hours)",1.0,12.0,7.0)
stress = st.selectbox("Stress Level",["Low","Medium","High"])
exercise = st.selectbox("Exercise Frequency",["None","Weekly","Daily"])
coffee = st.number_input("Daily Coffee Intake (cups)",0,10,1)
screen = st.number_input("Screen Time (hours)",0.0,12.0,4.0)

if st.button("Predict"):
    stress_enc = stress_map[stress]
    exercise_enc = exercise_map[exercise]

    user_data = [[sleep,stress_enc,exercise_enc,coffee,screen]]

    pred = model.predict(user_data)[0]

    result = list(quality_map.keys())[list(quality_map.values()).index(pred)]

    st.success(f"Predicted Sleep Quality : {result}")

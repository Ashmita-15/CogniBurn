import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("burnout_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="CogniBurn", layout="centered")

st.title("🧠 CogniBurn: Burnout Detection System")
st.write("Enter your lifestyle and behavioral data to predict burnout level.")

# 🔁 Mapping functions
level_map = {
    "Very Low": 0.0,
    "Low": 0.25,
    "Neutral": 0.5,
    "High": 0.75,
    "Very High": 1.0
}

quality_map = {
    "Awful": 0.0,
    "Bad": 0.25,
    "Medium": 0.5,
    "Good": 0.75,
    "Excellent": 1.0
}

# 🧾 INPUTS (REAL WORLD)

study_time = st.number_input("Study Time (hours/day)", 0.0, 12.0, 4.0)
social_media_usage = st.number_input("Social Media Usage (hours/day)", 0.0, 10.0, 2.0)
sleep_hours = st.number_input("Sleep Hours (per day)", 0.0, 12.0, 7.0)
physical_activity = st.number_input("Physical Activity (hours/week)", 0.0, 20.0, 3.0)

sleep_quality = st.selectbox("Sleep Quality", list(quality_map.keys()))
headache_freq = st.selectbox("Headache Frequency", list(quality_map.keys()))

study_load = st.selectbox("Study Load", list(level_map.keys()))
exam_anxiety = st.selectbox("Exam Anxiety", list(level_map.keys()))
stress_level = st.selectbox("Stress Level", list(level_map.keys()))
mental_strain = st.selectbox("Mental Strain", list(level_map.keys()))
recovery_score = st.selectbox("Recovery Score", list(level_map.keys()))
performance_pressure = st.selectbox("Performance Pressure", list(level_map.keys()))
lifestyle_stress = st.selectbox("Lifestyle Stress", list(level_map.keys()))

screen_time = social_media_usage  # optional approximation
extracurricular_freq = 0.5  # default (can improve later)

# 🔄 NORMALIZATION (MATCH TRAINING DATA)

study_time = study_time / 12
social_media_usage = social_media_usage / 10
sleep_hours = sleep_hours / 12
physical_activity = physical_activity / 20

sleep_quality = quality_map[sleep_quality]
headache_freq = quality_map[headache_freq]

study_load = level_map[study_load]
exam_anxiety = level_map[exam_anxiety]
stress_level = level_map[stress_level]
mental_strain = level_map[mental_strain]
recovery_score = level_map[recovery_score]
performance_pressure = level_map[performance_pressure]
lifestyle_stress = level_map[lifestyle_stress]

# 🚀 PREDICTION
if st.button("Predict Burnout Level"):

    input_data = np.array([[study_time, social_media_usage, sleep_quality,
                            headache_freq, study_load, extracurricular_freq,
                            screen_time, sleep_hours, physical_activity,
                            exam_anxiety, stress_level, mental_strain,
                            recovery_score, performance_pressure,
                            lifestyle_stress]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    levels = {0: "Low", 1: "Medium", 2: "High"}
    result = levels[prediction]

    st.subheader(f"🔥 Burnout Level: {result}")

    # 💡 Recommendations
    st.write("### 💡 Recommendations:")

    if result == "High":
        st.error("⚠️ High burnout detected! Reduce workload, improve sleep, and take breaks.")
    elif result == "Medium":
        st.warning("⚠️ Moderate burnout. Maintain balance and monitor stress.")
    else:
        st.success("✅ Low burnout. Keep up the healthy routine!")
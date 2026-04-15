import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import pandas as pd

# Load model and scaler
model = joblib.load("burnout_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="CogniBurn", layout="centered")

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}

h1 {
    text-align: center;
}

.stButton>button {
    background-color: #6366f1;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}

.low { background-color: #16a34a; }
.medium { background-color: #f59e0b; }
.high { background-color: #dc2626; }

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1>🧠 CogniBurn</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-Based Burnout Detection System</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# MAPPINGS
# -------------------------------
level_map = {"Very Low": 0.0, "Low": 0.25, "Neutral": 0.5, "High": 0.75, "Very High": 1.0}
quality_map = {"Awful": 0.0, "Bad": 0.25, "Medium": 0.5, "Good": 0.75, "Excellent": 1.0}

# -------------------------------
# INPUTS
# -------------------------------
st.subheader("📊 Lifestyle Inputs")

col1, col2 = st.columns(2)

with col1:
    study_time = st.number_input("Study Time (hrs/day)", 0.0, 12.0, 4.0)
    sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
    physical_activity = st.number_input("Physical Activity (hrs/week)", 0.0, 20.0, 3.0)

with col2:
    social_media_usage = st.number_input("Social Media (hrs/day)", 0.0, 10.0, 2.0)
    screen_time = social_media_usage

st.subheader("😴 Health & Quality")

col1, col2 = st.columns(2)

with col1:
    sleep_quality = st.selectbox("Sleep Quality", list(quality_map.keys()))
    headache_freq = st.selectbox("Headache Frequency", list(quality_map.keys()))

with col2:
    study_load = st.selectbox("Study Load", list(level_map.keys()))
    extracurricular_freq = 0.5

st.subheader("🧠 Mental State")

col1, col2 = st.columns(2)

with col1:
    stress_level = st.selectbox("Stress Level", list(level_map.keys()))
    exam_anxiety = st.selectbox("Exam Anxiety", list(level_map.keys()))
    recovery_score = st.selectbox("Recovery Score", list(level_map.keys()))

with col2:
    mental_strain = st.selectbox("Mental Strain", list(level_map.keys()))
    performance_pressure = st.selectbox("Performance Pressure", list(level_map.keys()))
    lifestyle_stress = st.selectbox("Lifestyle Stress", list(level_map.keys()))

# -------------------------------
# NORMALIZATION
# -------------------------------
study_time /= 12
social_media_usage /= 10
sleep_hours /= 12
physical_activity /= 20

sleep_quality = quality_map[sleep_quality]
headache_freq = quality_map[headache_freq]

study_load = level_map[study_load]
exam_anxiety = level_map[exam_anxiety]
stress_level = level_map[stress_level]
mental_strain = level_map[mental_strain]
recovery_score = level_map[recovery_score]
performance_pressure = level_map[performance_pressure]
lifestyle_stress = level_map[lifestyle_stress]

# -------------------------------
# PREDICT
# -------------------------------
if st.button("🔍 Predict Burnout Level"):

    input_data = np.array([[study_time, social_media_usage, sleep_quality,
                            headache_freq, study_load, extracurricular_freq,
                            screen_time, sleep_hours, physical_activity,
                            exam_anxiety, stress_level, mental_strain,
                            recovery_score, performance_pressure,
                            lifestyle_stress]])

    input_scaled = scaler.transform(input_data)

    probs = model.predict_proba(input_scaled)[0]
    prediction = np.argmax(probs)
    confidence = np.max(probs) * 100

    levels = {0: "Low", 1: "Medium", 2: "High"}
    result = levels[prediction]

    # -------------------------------
    # RESULT BOX
    # -------------------------------
    if result == "Low":
        st.markdown("<div class='result-box low'>✅ Low Burnout</div>", unsafe_allow_html=True)
    elif result == "Medium":
        st.markdown("<div class='result-box medium'>⚠️ Medium Burnout</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box high'>🔥 High Burnout</div>", unsafe_allow_html=True)

    # -------------------------------
    # GAUGE CHART
    # -------------------------------
    st.markdown("### 📊 Burnout Risk Score")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 30], 'color': "#16a34a"},
                {'range': [30, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#dc2626"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------
    importance = np.mean(np.abs(model.coef_), axis=0)

    features = [
        "Study Time", "Social Media", "Sleep Quality", "Headache",
        "Study Load", "Extracurricular", "Screen Time", "Sleep Hours",
        "Physical Activity", "Exam Anxiety", "Stress", "Mental Strain",
        "Recovery", "Performance Pressure", "Lifestyle Stress"
    ]

    df_imp = pd.DataFrame({"Feature": features, "Importance": importance})
    st.markdown("### 📈 Feature Importance")
    st.bar_chart(df_imp.set_index("Feature"))

    # -------------------------------
    # INPUT VISUALIZATION
    # -------------------------------
    st.markdown("### 🧾 Your Input Summary")

    input_df = pd.DataFrame({
        "Feature": features,
        "Value": input_data.flatten()
    })

    st.bar_chart(input_df.set_index("Feature"))

    # -------------------------------
    # SMART RECOMMENDATIONS
    # -------------------------------
    st.markdown("### 🎯 Personalized Insights")

    if stress_level > 0.7:
        st.warning("High stress detected → Try relaxation techniques")

    if sleep_hours < 0.4:
        st.warning("Low sleep → Increase rest duration")

    if social_media_usage > 0.6:
        st.warning("High screen time → Reduce digital exposure")

    if result == "Low":
        st.success("Great balance! Keep it up 🎉")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with ❤️ using Machine Learning</p>", unsafe_allow_html=True)
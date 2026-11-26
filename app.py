"""
app.py - Streamlit UI for Student Mental Health Score Prediction.

Run:
    streamlit run app.py
"""

import streamlit as st

from pipeline import load_bundle, predict_score

st.set_page_config(
    page_title="Student Mental Health Score",
    page_icon="🧠",
    layout="centered",
)


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()
opts = bundle["options"]
rng = bundle["ranges"]
metrics = bundle["metrics"]

st.title("🧠 Student Mental Health Score Predictor")
st.caption(
    "Educational tool only. The prediction is not a clinical diagnosis "
    "or medical assessment."
)

with st.form("predict_form"):
    st.subheader("Demographics")
    c1, c2 = st.columns(2)
    age = c1.number_input(
        "Age", min_value=15, max_value=40, value=int(rng["Age"][2]), step=1
    )
    gender = c2.selectbox("Gender", opts["Gender"])

    c3, c4 = st.columns(2)
    country = c3.selectbox("Country", opts["Grouped_country"])
    academic = c4.selectbox("Academic Level", opts["Academic_Level"])

    st.subheader("Social Media Usage")
    c5, c6 = st.columns(2)
    platform = c5.selectbox("Most Used Platform", opts["Most_Used_Platform"])
    purpose = c6.selectbox("Purpose of Use", opts["Purpose_Of_Use"])

    usage = st.slider(
        "Average Daily Usage (hours)",
        0.0, 16.0, float(rng["Avg_Daily_Usage_Hours"][2]), 0.1,
    )
    unlocks = st.number_input(
        "Daily Phone Unlocks",
        min_value=0, max_value=600,
        value=int(rng["Daily_Unlocks"][2]), step=1,
    )

    st.subheader("Lifestyle")
    study = st.slider(
        "Study Hours (per day)", 0.0, 16.0, float(rng["Study_Hours"][2]), 0.1
    )
    activity = st.slider(
        "Physical Activity (hours per day)",
        0.0, 8.0, float(rng["Physical_Activity_Hours"][2]), 0.1,
    )
    sleep = st.slider(
        "Sleep (hours per night)",
        2.0, 12.0, float(rng["Sleep_Hours_Per_Night"][2]), 0.1,
    )
    stress = st.select_slider(
        "Stress Level", options=opts["Stress_Level"], value="Medium"
    )

    submitted = st.form_submit_button("Predict Mental Health Score", type="primary")

if submitted:
    user_input = {
        "Age": age,
        "Gender": gender,
        "Country": country,
        "Academic_Level": academic,
        "Most_Used_Platform": platform,
        "Purpose_Of_Use": purpose,
        "Avg_Daily_Usage_Hours": usage,
        "Daily_Unlocks": unlocks,
        "Study_Hours": study,
        "Physical_Activity_Hours": activity,
        "Sleep_Hours_Per_Night": sleep,
        "Stress_Level": stress,
    }
    score = predict_score(bundle, user_input)

    st.divider()
    st.subheader("Result")
    st.metric("Predicted Mental Health Score (0-10)", f"{score:.2f}")
    st.progress(min(max(score / 10, 0.0), 1.0))

    mae = metrics["mae"]
    st.caption(
        f"Typical model error is about ±{mae:.2f} points, "
        f"so the likely range is {max(score - mae, 0):.1f} to "
        f"{min(score + mae, 10):.1f}."
    )

    if score >= 7.5:
        st.success("Higher score: indicators suggest good well-being.")
    elif score >= 5.5:
        st.info("Moderate score: a few habits may be worth improving.")
    else:
        st.warning(
            "Lower score: consider reducing screen time, improving sleep, "
            "and talking to someone you trust or a counsellor."
        )

with st.expander("Model information"):
    st.write("**Model:** Random Forest Regressor (tuned with RandomizedSearchCV)")
    st.write(f"**Test R²:** {metrics['r2_test']:.3f}")
    st.write(f"**Test MAE:** {metrics['mae']:.3f}")
    st.write(f"**Test RMSE:** {metrics['rmse']:.3f}")

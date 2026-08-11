
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

try:
    model = joblib.load("disease_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    features = joblib.load("features.pkl")
except Exception as e:
    st.error("Model files could not be loaded.")
    st.error(str(e))
    st.stop()


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f9ff;
}

.title {
    text-align: center;
    color: #1565c0;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #e8f5e9;
    border: 2px solid #43a047;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    color: #2e7d32;
    font-size: 28px;
    font-weight: bold;
}

.symptom-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="title">🩺 Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Select your symptoms to predict a possible disease</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🩺 Disease Predictor")

st.sidebar.info(
    """
    Select the symptoms you are experiencing
    and click **Predict Disease**.
    
    This application is for educational purposes
    and should not replace professional medical advice.
    """
)

# ==========================================
# SYMPTOM SELECTION
# ==========================================

st.subheader("🔍 Select Symptoms")

st.write("Choose the symptoms you are currently experiencing:")

# Create columns
col1, col2, col3 = st.columns(3)

selected_symptoms = []

# Divide symptoms into 3 columns
total_features = len(features)
third = (total_features + 2) // 3

feature_groups = [
    features[:third],
    features[third:2 * third],
    features[2 * third:]
]

with col1:
    st.markdown("### Symptoms")
    for symptom in feature_groups[0]:
        label = symptom.replace("_", " ").title()

        if st.checkbox(label, key=f"symptom_{symptom}"):
            selected_symptoms.append(symptom)

with col2:
    st.markdown("### Symptoms")
    for symptom in feature_groups[1]:
        label = symptom.replace("_", " ").title()

        if st.checkbox(label, key=f"symptom_{symptom}"):
            selected_symptoms.append(symptom)

with col3:
    st.markdown("### Symptoms")
    for symptom in feature_groups[2]:
        label = symptom.replace("_", " ").title()

        if st.checkbox(label, key=f"symptom_{symptom}"):
            selected_symptoms.append(symptom)


# ==========================================
# SELECTED SYMPTOMS
# ==========================================

st.divider()

if selected_symptoms:
    st.subheader("Selected Symptoms")

    display_symptoms = [
        symptom.replace("_", " ").title()
        for symptom in selected_symptoms
    ]

    st.write(", ".join(display_symptoms))

else:
    st.info("No symptoms selected yet.")


# ==========================================
# PREDICTION
# ==========================================

st.divider()

if st.button("🔮 Predict Disease", use_container_width=True):

    if len(selected_symptoms) == 0:

        st.warning(
            "⚠️ Please select at least one symptom before predicting."
        )

    else:

        # Create input dataframe
        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=features
        )

        # Set selected symptoms to 1
        for symptom in selected_symptoms:
            input_data[symptom] = 1

        # Prediction
        prediction = model.predict(input_data)

        # Convert encoded value to disease name
        disease = label_encoder.inverse_transform(prediction)[0]

        # Probability if supported by model
        try:
            probabilities = model.predict_proba(input_data)[0]

            confidence = max(probabilities) * 100

        except Exception:
            confidence = None

        # ==========================================
        # RESULT
        # ==========================================

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-title">
                    🩺 Predicted Disease
                </div>
                <h2>{disease}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        if confidence is not None:
            st.success(
                f"Prediction confidence: {confidence:.2f}%"
            )

        st.warning(
            "⚠️ This prediction is for educational purposes only. "
            "Please consult a qualified healthcare professional "
            "for proper diagnosis and treatment."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Disease Prediction System | Machine Learning + Streamlit"
)

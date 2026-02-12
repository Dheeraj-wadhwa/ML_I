
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# Page Config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)

# Load Model and Data
@st.cache_resource
def load_resources():
    model = joblib.load('breast_cancer_pipeline.pkl')
    data = load_breast_cancer()
    feature_names = data.feature_names
    X = pd.DataFrame(data.data, columns=feature_names)
    y = data.target
    target_names = data.target_names
    return model, feature_names, X, y, target_names

try:
    model, feature_names, X, y, target_names = load_resources()
except FileNotFoundError:
    st.error("Error: 'breast_cancer_pipeline.pkl' not found. Please run the training script first.")
    st.stop()

# Title and Description
st.title("🎗️ Breast Cancer Prediction System")
st.markdown("""
This application uses a Machine Learning model to predict whether a breast mass is **Malignant** or **Benign** 
based on 30 measurements extracted from digitized images of fine needle aspirate (FNA).
""")

# Sidebar
st.sidebar.header("Input Features")

# Helper function to get random sample
def get_random_sample():
    random_index = np.random.randint(0, len(X))
    return X.iloc[random_index], y[random_index]

# Initialize session state for input values if not exists
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = X.mean() # Default to mean values
    st.session_state['true_label'] = None

# Buttons to load data
if st.sidebar.button("🎲 Load Random Sample"):
    sample_X, sample_y = get_random_sample()
    st.session_state['input_data'] = sample_X
    st.session_state['true_label'] = sample_y

# Create inputs for all 30 features
input_dict = {}
st.sidebar.markdown("---")
st.sidebar.write("### Adjust Measurements")

# Group features for better UI (optional, but good for structure)
# We'll just list them in order for now as there are many
for feature in feature_names:
    val = float(st.session_state['input_data'][feature])
    # Create a slider/number input. Using number_input for precision.
    input_dict[feature] = st.sidebar.number_input(
        label=feature.replace('_', ' ').capitalize(),
        value=val,
        format="%.4f"
    )

# Convert input to DataFrame
input_df = pd.DataFrame([input_dict])

# Main Display
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Measurements")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), height=400)

with col2:
    st.subheader("Prediction Result")
    
    if st.button("🚀 Analyze Features", type="primary"):
        # Prediction
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0]
        
        class_name = target_names[prediction]
        probability = prediction_prob[prediction]

        # Display Logic
        if prediction == 0: # Malignant
            st.error(f"## Prediction: {class_name.upper()}")
            st.metric("Confidence", f"{probability:.2%}")
            st.markdown("⚠️ **High Risk**: Immediate consultation recommended.")
        else: # Benign
            st.success(f"## Prediction: {class_name.upper()}")
            st.metric("Confidence", f"{probability:.2%}")
            st.markdown("✅ **Low Risk**: Likely benign, but standard follow-up advised.")

    # Show True Label if available (from Random Sample)
    if st.session_state['true_label'] is not None:
        st.markdown("---")
        true_class = target_names[st.session_state['true_label']]
        st.write(f"**True Label (from dataset):** `{true_class}`")

# Footer
st.markdown("---")
st.markdown(f"*Model trained on {len(X)} samples with 30 features. Accuracy: >95%.*")


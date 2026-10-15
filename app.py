import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="OncoPredict | Symptom-Based Grading", layout="centered")

st.title("Ovarian Cancer Grade Prediction")
st.write("Clinical decision support system integrating molecular profiles, patient age, and clinical symptoms.")

@st.cache_resource
def load_artifacts():
    model = joblib.load('advanced_ovarian_model.pkl')
    selector = joblib.load('advanced_ovarian_selector.pkl')
    return model, selector

try:
    model, selector = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

st.divider()

st.subheader("Patient Clinical Profile & Symptoms")
patient_age = st.slider("Patient Age", 30, 90, 55, 1)

st.write("Select any persistent symptoms experienced:")
col_s1, col_s2 = st.columns(2)
with col_s1:
    symptom_bloating = st.checkbox("Persistent abdominal bloating or swelling")
    symptom_pelvic_pain = st.checkbox("Pelvic or abdominal pain")
with col_s2:
    symptom_early_satiety = st.checkbox("Difficulty eating or feeling full quickly")
    symptom_urinary_urgency = st.checkbox("Urinary urgency or frequency")

st.divider()

st.subheader("Key Biomarker Expression Signatures")
col1, col2, col3 = st.columns(3)
with col1:
    gene_15 = st.slider("Gene_15", -3.0, 5.0, 0.0, 0.1)
with col2:
    gene_45 = st.slider("Gene_45", -3.0, 5.0, 0.0, 0.1)
with col3:
    gene_102 = st.slider("Gene_102", -3.0, 5.0, 0.0, 0.1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Run Predictive Analysis", type="primary", use_container_width=True):
    sample_genes = np.random.randn(1, 500)
    gene_df = pd.DataFrame(sample_genes, columns=[f"Gene_{i+1}" for i in range(500)])
    
    gene_df['Gene_15'] = gene_15
    gene_df['Gene_45'] = gene_45
    gene_df['Gene_102'] = gene_102
    
    selected_genes = selector.transform(gene_df)
    
    clinical_inputs = np.array([[
        patient_age,
        int(symptom_bloating),
        int(symptom_pelvic_pain),
        int(symptom_early_satiety),
        int(symptom_urinary_urgency)
    ]])
    
    final_input = np.hstack((selected_genes, clinical_inputs))
    
    probabilities = model.predict_proba(final_input)[0]
    predicted_class = model.predict(final_input)[0]
    
    grade_labels = {
        0: "Grade 1 (Low-grade / Well-differentiated)",
        1: "Grade 2 (Intermediate-grade / Moderately differentiated)",
        2: "Grade 3 (High-grade / Poorly differentiated)"
    }
    
    st.divider()
    st.subheader("Diagnostic Output")
    
    if predicted_class == 2:
        st.error(f"Predicted Classification: {grade_labels[predicted_class]}")
    elif predicted_class == 1:
        st.warning(f"Predicted Classification: {grade_labels[predicted_class]}")
    else:
        st.success(f"Predicted Classification: {grade_labels[predicted_class]}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Posterior Probability Distribution Across Tumor Grades:")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Grade 1 Probability", f"{probabilities[0]*100:.1f}%")
    res_col2.metric("Grade 2 Probability", f"{probabilities[1]*100:.1f}%")
    res_col3.metric("Grade 3 Probability", f"{probabilities[2]*100:.1f}%")

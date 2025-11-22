import streamlit as st
import joblib
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Root path (depuis app/)

from utils.processing import ClinicalPrediction

# Config
st.set_page_config(page_title="Diagnostic Clinique IA", layout="wide")
MODEL_PATH = '../models/model.pkl'  # Relatif depuis app/

# Titre
st.title("🩺 Diagnostic d'Infection - Interface Clinique")
st.write("Saisissez les données du patient pour un diagnostic instantané.")

# Charger modèle
@st.cache_resource
def load_model():
    try:
        saved = joblib.load(MODEL_PATH)
        return saved['predictor']
    except FileNotFoundError:
        st.error("Modèle manquant. Relance train.py.")
        st.stop()

predictor = load_model()

# Sidebar pour inputs
st.sidebar.header("Données Patient")
age = st.sidebar.slider("Âge (18-80)", 18, 80, 45)
temperature = st.sidebar.slider("Température (°C)", 36.0, 40.0, 37.0, 0.1)
symptoms = st.sidebar.slider("Score Symptômes (0-10)", 0, 10, 3)
fatigue = st.sidebar.selectbox("Fatigue (0: Non, 1: Oui)", [0, 1])

# Bouton diagnostic
if st.sidebar.button("Diagnostiquer"):
    patient_data = np.array([[age, temperature, symptoms, fatigue]])
    result = predictor.diagnose(patient_data)
    proba = predictor.model.predict_proba(patient_data)[0][1]
    
    # Affichage principal
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Résultat")
        if result == "infecté":
            st.error(f"🚨 **{result.upper()}** - Consultez un médecin !")
        else:
            st.success(f"✅ **{result.upper()}** - Surveillance recommandée.")
    
    with col2:
        st.subheader("Probabilité")
        st.metric("Risque d'Infection", f"{proba:.1%}")
    
    # Explications
    st.write("**Interprétation :** Basé sur un modèle ML. Seuil >50% = infecté.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Modèle sur données synthétiques. Pour prod, vraies données.")
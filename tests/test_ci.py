import sys
import os
import joblib
import numpy as np
import pytest

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.processing import ClinicalPrediction

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/model.pkl')

def test_model_loading():
    """Test that the model file exists and can be loaded."""
    assert os.path.exists(MODEL_PATH), "Model file not found"
    saved = joblib.load(MODEL_PATH)
    assert 'predictor' in saved, "Key 'predictor' missing in saved model"

def test_prediction_flow():
    """Test a simple prediction flow."""
    saved = joblib.load(MODEL_PATH)
    predictor = saved['predictor']
    
    # Dummy patient data: [age, temperature, symptoms, fatigue]
    patient_data = np.array([[45, 37.5, 5, 1]])
    
    # We expect a string result (sain or infecté)
    result = predictor.diagnose(patient_data)
    assert result in ["infecté", "sain"], f"Unexpected result: {result}"

import argparse
import joblib
import numpy as np
import os
import sys  # AJOUT : Pour le path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # AJOUT : Ajoute le root au path

from core.dataset import ClinicalDataset
from utils.processing import ClinicalPrediction
from utils.metrics import evaluate_model

def main(model_path='models/model.pkl'):
    # Charger
    saved = joblib.load(model_path)
    model_obj = saved['model_obj']
    predictor = saved['predictor']
    
    # Dataset pour test
    dataset = ClinicalDataset()
    dataset.load_csv('data/patient_data.csv')
    _, X_test, _, y_test = dataset.split()
    
    # Prédictions
    y_pred = model_obj.predict(X_test)
    y_pred_proba = model_obj.predict_proba(X_test)
    
    # Évaluer
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    print(f"Évaluation {model_obj.model_type} : {metrics}")
    
    # Exemple
    patient_example = np.array([[45, 38.5, 7, 1]])
    result = predictor.diagnose(patient_example)
    proba = model_obj.predict_proba(patient_example)[0][1]
    print(f"Exemple : {result} (proba: {proba:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/model.pkl', help='Chemin modèle')
    args = parser.parse_args()
    main(args.model)
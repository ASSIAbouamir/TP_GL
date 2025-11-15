import argparse
import joblib
import os
import sys  # Pour le path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Ajoute le root au path

from core.dataset import ClinicalDataset
from core.model import ClinicalModel
from utils.processing import ClinicalPrediction
from utils.metrics import evaluate_model

def main(data_path='data/patient_data.csv', model_path='models/model.pkl', model_type='rf'):
    # Créer dossiers (gestion robuste pour dirname vide)
    os.makedirs('data', exist_ok=True)
    model_dir = os.path.dirname(model_path)
    if model_dir:  # Si dirname n'est pas vide
        os.makedirs(model_dir, exist_ok=True)
    
    # Dataset
    dataset = ClinicalDataset()
    if not os.path.exists(data_path):
        print("Génération synthétique...")
        dataset.generate_synthetic()
        dataset.save_csv(data_path)
    else:
        dataset.load_csv(data_path)
    
    # Split
    X_train, X_test, y_train, y_test = dataset.split()
    
    # Modèle
    model_obj = ClinicalModel(model_type=model_type)
    model_obj.train(X_train, y_train)
    
    # Évaluer
    y_pred = model_obj.predict(X_test)
    y_pred_proba = model_obj.predict_proba(X_test)
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    print(f"Précision : {model_obj.evaluate(X_test, y_test):.2f}")
    
    # Wrapper
    predictor = ClinicalPrediction(model_obj.model)  # Utilise le modèle entraîné
    
    # Sauvegarde
    joblib.dump({
        'model_obj': model_obj,  # Inclut type et params
        'predictor': predictor,
        'metrics': metrics
    }, model_path)
    print(f"Modèle {model_type} sauvegardé : {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/patient_data.csv', help='Chemin données')
    parser.add_argument('--model', default='models/model.pkl', help='Chemin modèle')  # Changé par défaut
    parser.add_argument('--type', default='rf', choices=['rf', 'lr'], help='Type modèle')
    args = parser.parse_args()
    main(args.data, args.model, args.type)
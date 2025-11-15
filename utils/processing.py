import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class ClinicalPrediction:
    def __init__(self, model):
        """
        Initialise la classe avec un modèle déjà entraîné.
        
        :param model: Modèle entraîné (par exemple, de scikit-learn) pour la prédiction binaire.
        """
        self.model = model
        self.scaler = StandardScaler()  # Pour scaler les données si besoin
    
    def preprocess(self, data):
        """
        Préprocessing simple : scale les features numériques.
        
        :param data: DataFrame ou array-like.
        :return: Données scalées.
        """
        if isinstance(data, pd.DataFrame):
            X = data.drop('target', axis=1, errors='ignore').values
        else:
            X = data
        return self.scaler.fit_transform(X) if not hasattr(self.scaler, 'scale_') else self.scaler.transform(X)
    
    def diagnose(self, patient_data):
        """
        Diagnostique un patient en fonction des données fournies.
        
        :param patient_data: Données du patient (array-like de shape (1, n_features)).
        :return: 'infecté' si la probabilité > 0.5, sinon 'sain'.
        """
        patient_data = self.preprocess(patient_data.reshape(1, -1))
        proba = self.model.predict_proba(patient_data)[0][1]
        
        if proba > 0.5:
            return "infecté"
        else:
            return "sain"

def load_and_split_data(file_path, test_size=0.2, random_state=42):
    """
    Charge les données CSV et les split en train/test.
    
    :param file_path: Chemin vers le CSV.
    :return: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(file_path)
    X = df.drop('target', axis=1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
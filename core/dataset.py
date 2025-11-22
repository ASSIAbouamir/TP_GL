import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ClinicalDataset:
    def __init__(self, n_samples=1000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.df = None
    
    def generate_synthetic(self):
        """Génère des données synthétiques pour prédiction d'infection."""
        np.random.seed(self.random_state)
        age = np.random.randint(18, 81, self.n_samples)
        temperature = np.random.normal(37, 1, self.n_samples)
        symptoms = np.random.randint(0, 11, self.n_samples)
        fatigue = np.random.binomial(1, 0.5, self.n_samples)
        X = np.column_stack([age, temperature, symptoms, fatigue])
        y = np.where((temperature > 38) & (symptoms > 5), 1, 0)
        
        self.df = pd.DataFrame(X, columns=['age', 'temperature', 'symptoms', 'fatigue'])
        self.df['target'] = y
        return self.df
    
    def load_csv(self, file_path):
        """Charge un CSV existant."""
        self.df = pd.read_csv(file_path)
        return self.df
    
    def split(self, test_size=0.2):
        """Split en train/test."""
        if self.df is None:
            raise ValueError("Générez ou chargez d'abord les données.")
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
    
    def save_csv(self, file_path):
        """Sauvegarde en CSV."""
        if self.df is not None:
            self.df.to_csv(file_path, index=False)
            print(f"Dataset sauvegardé : {file_path}")
"""Script pour générer le dataset synthétique."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Ajoute le root au path

from core.dataset import ClinicalDataset

if __name__ == "__main__":
    dataset = ClinicalDataset(n_samples=1000)
    df = dataset.generate_synthetic()
    dataset.save_csv('data/patient_data.csv')
    print("Données générées et sauvegardées dans data/patient_data.csv")
    print(f"Shape : {df.shape}")
    print(df.head())
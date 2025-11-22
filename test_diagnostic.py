import joblib
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))  # Root path

from utils.processing import ClinicalPrediction

# Charge le modèle
MODEL_PATH = 'models/model.pkl'
try:
    saved = joblib.load(MODEL_PATH)
    predictor = saved['predictor']
    model = predictor.model
    print("✅ Modèle chargé !")
except FileNotFoundError:
    print("❌ Modèle manquant. Relance train.py.")
    exit()

print("\n🩺 Test Diagnostic Console")
print("Saisis données patient (ou 'q' pour quitter)")
print("-" * 50)

while True:
    try:
        age_input = input("\nÂge (18-80, ou 'q') : ").strip()
        if age_input.lower() == 'q':
            break
        age = max(18, min(80, int(age_input or 45)))

        temp_input = input("Température (°C) : ").strip() or "37.0"
        temperature = float(temp_input)

        symptoms = int(input("Symptômes (0-10) : ") or "3")
        fatigue = int(input("Fatigue (0/1) : ") or "0")

        # Prédiction
        patient_data = np.array([[age, temperature, symptoms, fatigue]])
        result = predictor.diagnose(patient_data)
        proba = model.predict_proba(patient_data)[0][1]

        # Résultat
        print("\n" + "=" * 50)
        print("DIAGNOSTIC :")
        if result == "infecté":
            print("🚨 **INFECTÉ** - Médecin !")
        else:
            print("✅ **SAIN** - OK.")
        print(f"Proba infection : {proba:.2%}")
        print(f"Données : Âge={age}, Temp={temperature}, Sympt={symptoms}, Fatigue={fatigue}")
        print("=" * 50)
    except ValueError:
        print("❌ Saisie invalide (utilise nombres).")
    except KeyboardInterrupt:
        print("\nBye !")
        break

print("Test OK. Projet prêt !")
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000
age = np.random.randint(18, 81, n_samples)
temperature = np.random.normal(37, 1, n_samples)
symptoms = np.random.randint(0, 11, n_samples)
fatigue = np.random.binomial(1, 0.5, n_samples)
X = np.column_stack([age, temperature, symptoms, fatigue])
y = np.where((temperature > 38) & (symptoms > 5), 1, 0)

df = pd.DataFrame(X, columns=['age', 'temperature', 'symptoms', 'fatigue'])
df['target'] = y
df.to_csv('data/patient_data.csv', index=False)
print("Fichier CSV généré !")
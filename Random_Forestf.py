import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

# -----------------------------
# 1. Charger dataset
# -----------------------------
df = pd.read_csv("drugvirusnet.csv")  
# colonnes attendues :
# virus_id | virus_sequence | drug_smiles | interaction (0/1)

# -----------------------------
# 2. Drug Fingerprint
# -----------------------------
def drug_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

df["drug_fp"] = df["drug_smiles"].apply(lambda x: np.array(drug_to_fp(x)))

# -----------------------------
# 3. Virus Sequence Encoding (AAC)
# -----------------------------
def aa_composition(seq):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    comps = []
    for aa in amino_acids:
        comps.append(seq.count(aa) / len(seq))
    return np.array(comps)

df["virus_vec"] = df["virus_sequence"].apply(lambda x: aa_composition(x))

# -----------------------------
# 4. Fusion features
# -----------------------------
X = np.stack(df.apply(lambda r: np.concatenate([r["drug_fp"], r["virus_vec"]]), axis=1))
y = df["interaction"].values

# -----------------------------
# 5. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 6. Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluation
# -----------------------------
pred = model.predict_proba(X_test)[:,1]

print("AUC:", roc_auc_score(y_test, pred))
print(classification_report(y_test, pred > 0.5))

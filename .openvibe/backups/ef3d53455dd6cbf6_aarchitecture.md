# 📐 Architecture et Pipeline Complet du Projet de Génération d'Écriture Manuscrite

## 🎯 Vue d'ensemble du Projet

Ce projet implémente un **système de génération d'écriture manuscrite** à partir de texte, utilisant deux approches principales :
1. **GAN Conditionnel (cGAN)** : Génération d'images d'écriture manuscrite via un réseau antagoniste génératif
2. **Rendu basé sur polices** : Génération stylisée utilisant des polices manuscrites avec effets réalistes

---

## 🏗️ Architecture Globale

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLET DU PROJET                   │
└─────────────────────────────────────────────────────────────────┘

1. PRÉPARATION DES DONNÉES
   │
   ├─ Dataset IAM (Images + Strokes + Transcriptions)
   │
   ├─ prepare_data.py
   │  ├─ Extraction des strokes (traits) depuis XML
   │  ├─ Extraction des transcriptions ASCII
   │  ├─ Normalisation et préprocessing
   │  └─ Sauvegarde en format NumPy (.npy)
   │
   └─ data/processed/
      ├─ x.npy      (strokes: offsets dx, dy, eos)
      ├─ x_len.npy  (longueurs réelles)
      ├─ c.npy      (transcriptions encodées)
      ├─ c_len.npy  (longueurs de texte)
      └─ w_id.npy   (IDs des écrivains)

2. ENTRÂINEMENT DU GAN
   │
   ├─ GAN/dataset.py (IAMDataset)
   │  └─ Conversion strokes → images 128x128
   │
   ├─ GAN/model.py
   │  ├─ Generator (ResNet-based)
   │  │  └─ Input: [bruit(100) + texte_embed(20×128)] → Image 128×128
   │  │
   │  └─ Discriminator (PatchGAN-like)
   │     └─ Input: [image(128×128) + texte_embed] → Score réel/faux
   │
   ├─ GAN/train.py
   │  ├─ Boucle d'entraînement adversarial
   │  ├─ Loss: MSE (LSGAN)
   │  └─ Sauvegarde checkpoints + samples
   │
   └─ GAN/checkpoints/
      ├─ generator_X.pth
      └─ discriminator_X.pth

3. GÉNÉRATION & INFÉRENCE
   │
   ├─ GAN/app.py (Streamlit)
   │  └─ Interface web pour génération avec GAN entraîné
   │
   └─ handwriting_renderer.py
      └─ Rendu stylisé basé sur polices (alternative au GAN)

4. ÉVALUATION
   │
   ├─ prepare_evaluation_data.py
   │  └─ Génère paires (réel, généré) pour métriques
   │
   ├─ metrics.py
   │  ├─ FID, KID (qualité visuelle)
   │  ├─ CER, WER (reconnaissance de texte)
   │  ├─ SSIM, PSNR, LPIPS (similarité)
   │  └─ OCR Accuracy
   │
   └─ evaluate_metrics.py
      └─ Script d'évaluation complète
```

---

## 📊 Pipeline Détaillé Étape par Étape

### **ÉTAPE 1 : Préparation des Données (`prepare_data.py`)**

#### 1.1 Vérification du Dataset IAM
```python
check_dataset_exists()
```
- Vérifie la présence des répertoires :
  - `data/ascii/` : Transcriptions textuelles
  - `data/lineStrokes/` : Fichiers de traits (strokes)
  - `data/original-xml/` : Métadonnées XML

#### 1.2 Collecte des Données
```python
collect_data()
```
**Processus :**
1. Parcourt récursivement `data/ascii/` pour trouver tous les fichiers `.txt`
2. Pour chaque fichier ASCII :
   - Extrait le texte (transcription)
   - Trouve le fichier XML correspondant dans `original-xml/`
   - Récupère l'ID de l'écrivain (`writerID`)
   - Trouve les fichiers de strokes correspondants dans `lineStrokes/`
3. Filtre les échantillons blacklistés (qualité faible)
4. Retourne : `(stroke_fnames, transcriptions, writer_ids)`

#### 1.3 Traitement des Strokes
```python
get_stroke_sequence(filename)
```
**Transformation :**
```
XML (coordonnées absolues)
  ↓
Coordonnées (x, y, eos)
  ↓ drawing.align()      → Correction de l'inclinaison
  ↓ drawing.denoise()     → Lissage Savitzky-Golay
  ↓ drawing.coords_to_offsets() → Conversion en déplacements
  ↓ drawing.normalize()   → Normalisation
  ↓
Offsets normalisés [dx, dy, eos] (MAX_STROKE_LEN=1200)
```

**Format des offsets :**
- `dx, dy` : Déplacements relatifs (normalisés)
- `eos` : End-of-stroke (1 = fin de trait, 0 = continuation)

#### 1.4 Traitement des Transcriptions
```python
get_ascii_sequences(filename)
```
**Processus :**
1. Lit le fichier ASCII
2. Extrait les lignes après `CSR:`
3. Encode chaque caractère en index dans `drawing.alphabet`
4. Tronque à `MAX_CHAR_LEN=75` caractères

#### 1.5 Sauvegarde
```python
# Tableaux NumPy créés
x = np.zeros([N, MAX_STROKE_LEN, 3])      # Strokes
x_len = np.zeros([N])                      # Longueurs réelles
c = np.zeros([N, MAX_CHAR_LEN])           # Transcriptions
c_len = np.zeros([N])                      # Longueurs de texte
w_id = np.zeros([N])                       # IDs écrivains

# Filtrage des échantillons valides
valid_mask = ~np.any(np.linalg.norm(x_i[:, :2], axis=1) > 60)

# Sauvegarde
np.save('data/processed/x.npy', x[valid_mask])
np.save('data/processed/x_len.npy', x_len[valid_mask])
np.save('data/processed/c.npy', c[valid_mask])
np.save('data/processed/c_len.npy', c_len[valid_mask])
np.save('data/processed/w_id.npy', w_id[valid_mask])
```

---

### **ÉTAPE 2 : Dataset PyTorch (`GAN/dataset.py`)**

#### 2.1 Chargement des Données
```python
IAMDataset(img_size=128, max_text_len=20)
```
- Charge les fichiers `.npy` depuis `data/processed/`
- Filtre les textes > `max_text_len` caractères

#### 2.2 Rendu Strokes → Image
```python
__getitem__(idx)
```

**Processus de conversion :**

1. **Récupération des strokes**
   ```python
   strokes = x[real_idx][:stroke_len]  # (L, 3) : [dx, dy, eos]
   ```

2. **Conversion offsets → coordonnées**
   ```python
   coords = np.cumsum(strokes[:, :2], axis=0)  # Accumulation des déplacements
   ```

3. **Normalisation et centrage**
   ```python
   # Calcul des min/max
   min_x, min_y = np.min(coords[:, 0]), np.min(coords[:, 1])
   max_x, max_y = np.max(coords[:, 0]), np.max(coords[:, 1])
   
   # Scaling pour tenir dans 128×128 avec padding
   scale = min(target_size / width, target_size / height)
   coords = (coords - [min_x, min_y]) * scale + padding
   ```

4. **Dessin avec PIL**
   ```python
   img = Image.new('L', (128, 128), color=255)  # Fond blanc
   draw = ImageDraw.Draw(img)
   
   # Dessine chaque trait (séparé par eos=1)
   for i in range(len(coords)):
       if coords[i, 2] == 1:  # End of stroke
           points = coords[start_idx:i+1, :2]
           draw.line(points, fill=0, width=2)  # Noir
   ```

5. **Transformation**
   ```python
   transform = transforms.Compose([
       transforms.ToTensor(),           # [0, 255] → [0, 1]
       transforms.Normalize((0.5,), (0.5,))  # [0, 1] → [-1, 1]
   ])
   ```

6. **Traitement du texte**
   ```python
   text = "".join([drawing.alphabet[i] for i in text_codes[:text_len]])
   text_indices = [char_to_idx.get(c, 0) for c in text]
   # Padding/truncation à max_text_len=20
   text_tensor = torch.tensor(text_indices, dtype=torch.long)
   ```

**Sortie :** `(img_tensor, text_tensor)`
- `img_tensor` : `(1, 128, 128)` dans `[-1, 1]`
- `text_tensor` : `(20,)` indices de caractères

---

### **ÉTAPE 3 : Architecture du GAN (`GAN/model.py`)**

#### 3.1 Generator (Générateur)

**Architecture :**

```
Input:
  - noise: (B, 100)          # Vecteur de bruit aléatoire
  - text_indices: (B, 20)     # Indices de caractères

1. Embedding du texte
   text_embed = Embedding(vocab_size, 128)(text_indices)
   → (B, 20, 128)
   
2. Flatten
   text_flat = text_embed.view(B, 20*128)
   → (B, 2560)
   
3. Concaténation
   x = concat([noise(100), text_flat(2560)])
   → (B, 2660)
   
4. Fully Connected
   fc = Linear(2660 → 512*4*4)
   → (B, 8192)
   reshape → (B, 512, 4, 4)
   
5. Upsampling progressif
   4×4 → 8×8  (Upsample + Conv + BN + ReLU)
   8×8 → 16×16
   16×16 → 32×32
   32×32 → 64×64
   64×64 → 128×128 (Final: Conv + Tanh)

Output:
  - gen_img: (B, 1, 128, 128) dans [-1, 1]
```

**Blocs ResNet :**
- Chaque bloc contient : `Conv → BN → ReLU → Conv → BN`
- Connexion résiduelle : `output = input + block(input)`

#### 3.2 Discriminator (Discriminateur)

**Architecture :**

```
Input:
  - img: (B, 1, 128, 128)
  - text_indices: (B, 20)

1. Traitement de l'image (downsampling)
   Conv2d(1 → 64, stride=2)   → (B, 64, 64)
   Conv2d(64 → 128, stride=2)  → (B, 128, 32)
   Conv2d(128 → 256, stride=2) → (B, 256, 16)
   Conv2d(256 → 512, stride=2)→ (B, 512, 8)
   
2. Traitement du texte
   text_embed = Embedding(vocab_size, 128)(text_indices)
   → (B, 20, 128)
   text_flat = text_embed.view(B, 2560)
   → (B, 2560)
   text_fc = Linear(2560 → 512*8*8)
   → (B, 32768)
   reshape → (B, 512, 8, 8)
   
3. Fusion
   combined = concat([img_features(512, 8, 8), text_features(512, 8, 8)])
   → (B, 1024, 8, 8)
   
4. Classification finale
   Conv2d(1024 → 512) → (B, 512, 8, 8)
   Conv2d(512 → 1)     → (B, 1, 4, 4)
   Average pooling     → (B, 1)
   Sigmoid             → Score [0, 1]

Output:
  - score: (B, 1)  # Probabilité que l'image soit réelle
```

---

### **ÉTAPE 4 : Entraînement (`GAN/train.py`)**

#### 4.1 Initialisation
```python
generator = Generator(vocab_size, text_embedding_dim=128, noise_dim=100, max_text_len=20)
discriminator = Discriminator(vocab_size, text_embedding_dim=128, max_text_len=20)

optimizer_G = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = MSELoss()  # LSGAN loss
```

#### 4.2 Boucle d'Entraînement

**Pour chaque batch :**

1. **Entraînement du Générateur**
   ```python
   # 1. Générer des images
   z = torch.randn(batch_size, 100)  # Bruit
   gen_imgs = generator(z, text_indices)
   
   # 2. Calculer la loss
   # Le générateur veut tromper le discriminateur
   g_loss = MSELoss(discriminator(gen_imgs, text_indices), ones)
   
   # 3. Backpropagation
   g_loss.backward()
   optimizer_G.step()
   ```

2. **Entraînement du Discriminateur**
   ```python
   # 1. Loss sur images réelles
   real_loss = MSELoss(discriminator(real_imgs, text_indices), ones)
   
   # 2. Loss sur images générées
   fake_loss = MSELoss(discriminator(gen_imgs.detach(), text_indices), zeros)
   
   # 3. Loss totale
   d_loss = 0.5 * (real_loss + fake_loss)
   
   # 4. Backpropagation
   d_loss.backward()
   optimizer_D.step()
   ```

3. **Sauvegarde**
   - Tous les 5 epochs : checkpoints
   - Chaque epoch : échantillons générés dans `GAN/samples/`

---

### **ÉTAPE 5 : Génération (Inference)**

#### 5.1 Avec le GAN (`GAN/app.py`)
```python
# 1. Charger le modèle entraîné
generator.load_state_dict(torch.load('checkpoint.pth'))

# 2. Préparer les inputs
text = "Hello World"
text_indices = [char_to_idx[c] for c in text]  # Padding à 20
z = torch.randn(1, 100)  # Bruit

# 3. Générer
with torch.no_grad():
    gen_img = generator(z, text_tensor)

# 4. Post-processing
img = (gen_img + 1) / 2.0  # [-1, 1] → [0, 1]
img = img * 255  # [0, 1] → [0, 255]
```

#### 5.2 Avec le Rendu Stylisé (`handwriting_renderer.py`)
```python
renderer = HandwritingRenderer(RenderConfig())

image = renderer.render(
    text="Hello World",
    font_name="Segoe Script",
    font_size=64,
    ink_color=(32, 32, 32),
    paper_style="plain",
    jitter_px=1.4,      # Tremblement
    tilt_degrees=-3.0,   # Inclinaison
    noise_strength=0.08, # Texture papier
    line_spacing=1.35
)
```

**Processus de rendu :**
1. Création d'une image blanche
2. Dessin du texte avec la police sélectionnée
3. Application du jitter (déplacement aléatoire des caractères)
4. Application de l'inclinaison (transformation affine)
5. Ajout d'ombre (Gaussian blur)
6. Ajout de bruit (texture papier)

---

### **ÉTAPE 6 : Évaluation (`metrics.py`, `evaluate_metrics.py`)**

#### 6.1 Préparation des Données d'Évaluation
```python
prepare_evaluation_data(num_samples=50)
```

**Génère deux ensembles :**
- `evaluation/real/` : Images rendues depuis les strokes réels
- `evaluation/gen/` : Images générées (GAN ou rendu stylisé)

#### 6.2 Métriques Calculées

**1. FID (Fréchet Inception Distance)**
- Mesure la distance entre distributions d'images réelles et générées
- Utilise Inception v3 pour extraire des features
- Plus bas = meilleur (typiquement < 50)

**2. KID (Kernel Inception Distance)**
- Version non-biaisée du FID
- Utilise un kernel polynomial
- Plus bas = meilleur

**3. CER (Character Error Rate)**
- Taux d'erreur au niveau des caractères
- Utilise la distance de Levenshtein
- 0.0 = parfait, 1.0 = toutes erreurs

**4. WER (Word Error Rate)**
- Taux d'erreur au niveau des mots
- 0.0 = parfait, 1.0 = toutes erreurs

**5. SSIM (Structural Similarity Index)**
- Similarité structurelle entre images
- 1.0 = identique, 0.0 = complètement différent

**6. PSNR (Peak Signal-to-Noise Ratio)**
- Ratio signal/bruit
- Plus haut = meilleur (typiquement 20-50 dB)

**7. LPIPS (Learned Perceptual Image Patch Similarity)**
- Similarité perceptuelle apprise
- Plus bas = meilleur (0.0 = identique)

**8. OCR Accuracy**
- Pourcentage de caractères correctement reconnus par OCR
- 1.0 = 100% correct

---

## 🔄 Flux de Données Complet

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUX DE DONNÉES                          │
└─────────────────────────────────────────────────────────────┘

1. DONNÉES BRUTES (IAM Dataset)
   │
   ├─ XML Files (strokes)
   │  └─ Coordonnées absolues (x, y, eos)
   │
   ├─ ASCII Files (transcriptions)
   │  └─ Texte brut
   │
   └─ Metadata (writer IDs)
      └─ Identifiants écrivains

2. PRÉTRAITEMENT (prepare_data.py)
   │
   ├─ Strokes
   │  └─ XML → Offsets normalisés [dx, dy, eos]
   │
   ├─ Textes
   │  └─ ASCII → Indices dans alphabet
   │
   └─ Sauvegarde
      └─ NumPy arrays (.npy)

3. DATASET PYTORCH (GAN/dataset.py)
   │
   ├─ Chargement .npy
   │
   ├─ Conversion strokes → images 128×128
   │  └─ PIL ImageDraw
   │
   └─ Transformation
      └─ Tensor + Normalisation [-1, 1]

4. ENTRAÎNEMENT (GAN/train.py)
   │
   ├─ Batch: (images, text_indices)
   │
   ├─ Generator
   │  └─ [noise + text] → image générée
   │
   ├─ Discriminator
   │  └─ [image + text] → score réel/faux
   │
   └─ Loss & Backprop
      └─ Mise à jour des poids

5. INFÉRENCE
   │
   ├─ GAN (GAN/app.py)
   │  └─ Texte → Image via modèle entraîné
   │
   └─ Rendu stylisé (handwriting_renderer.py)
      └─ Texte → Image via polices + effets

6. ÉVALUATION
   │
   ├─ Génération de paires (réel, généré)
   │
   ├─ Calcul métriques
   │  ├─ FID, KID (qualité visuelle)
   │  ├─ CER, WER (reconnaissance)
   │  └─ SSIM, PSNR, LPIPS (similarité)
   │
   └─ Rapport JSON
```

---

## 📁 Structure des Fichiers

```
GEN - Copie/
│
├── data/
│   ├── raw/                    # Dataset IAM brut
│   │   ├── ascii/
│   │   ├── lineStrokes/
│   │   └── original-xml/
│   │
│   └── processed/              # Données préprocessées
│       ├── x.npy               # Strokes
│       ├── x_len.npy           # Longueurs strokes
│       ├── c.npy               # Transcriptions
│       ├── c_len.npy           # Longueurs textes
│       └── w_id.npy            # IDs écrivains
│
├── GAN/
│   ├── model.py                # Generator + Discriminator
│   ├── dataset.py              # IAMDataset PyTorch
│   ├── train.py                # Script d'entraînement
│   ├── app.py                  # Interface Streamlit GAN
│   ├── checkpoints/            # Modèles entraînés
│   └── samples/                # Échantillons générés
│
├── evaluation/
│   ├── real/                   # Images réelles
│   └── gen/                   # Images générées
│
├── prepare_data.py             # Préparation dataset IAM
├── drawing.py                  # Utilitaires de rendu strokes
├── handwriting_renderer.py    # Rendu stylisé (polices)
├── metrics.py                  # Métriques d'évaluation
├── evaluate_metrics.py         # Script d'évaluation
├── prepare_evaluation_data.py  # Préparation données évaluation
├── check_data_rendering.py     # Vérification rendu
│
└── streamlit_app.py            # Interface principale (rendu stylisé)
```

---

## 🎯 Points Clés de l'Architecture

### 1. **Représentation des Strokes**
- Format : Offsets normalisés `[dx, dy, eos]`
- Avantages :
  - Invariant à la translation
  - Normalisé pour stabilité
  - Compact (1200 points max)

### 2. **Conditionnement du GAN**
- Le texte est embeddé et concaténé au bruit
- Le discriminateur reçoit aussi le texte
- Permet un contrôle précis de la génération

### 3. **Rendu On-the-Fly**
- Les strokes sont convertis en images à la volée dans le dataset
- Évite de stocker des milliers d'images
- Permet des transformations dynamiques

### 4. **Deux Approches Complémentaires**
- **GAN** : Apprentissage profond, style variable
- **Rendu stylisé** : Contrôle précis, rapide, pas d'entraînement

### 5. **Évaluation Multi-Métriques**
- Qualité visuelle (FID, KID)
- Reconnaissance (CER, WER, OCR)
- Similarité (SSIM, PSNR, LPIPS)

---

## 🚀 Workflow Typique

### **Entraînement d'un nouveau modèle :**
```bash
# 1. Préparer les données
python prepare_data.py

# 2. Vérifier le rendu
python check_data_rendering.py

# 3. Entraîner le GAN
cd GAN
python train.py --epochs 100 --batch_size 16

# 4. Générer des échantillons
# (automatique pendant l'entraînement)
```

### **Évaluation :**
```bash
# 1. Préparer les données d'évaluation
python prepare_evaluation_data.py

# 2. Calculer les métriques
python evaluate_metrics.py \
    --real_dir evaluation/real \
    --gen_dir evaluation/gen \
    --output metrics_results.json
```

### **Utilisation :**
```bash
# Interface Streamlit (rendu stylisé)
streamlit run streamlit_app.py

# Interface GAN (si modèle entraîné)
streamlit run GAN/app.py
```

---

## 📈 Hyperparamètres Principaux

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `image_size` | 128×128 | Taille des images générées |
| `latent_dim` | 100 | Dimension du vecteur de bruit |
| `text_embedding_dim` | 128 | Dimension de l'embedding texte |
| `max_text_len` | 20 | Longueur maximale du texte |
| `vocab_size` | ~70 | Taille de l'alphabet |
| `MAX_STROKE_LEN` | 1200 | Longueur max des séquences de strokes |
| `MAX_CHAR_LEN` | 75 | Longueur max des transcriptions |
| `learning_rate` | 0.0002 | Taux d'apprentissage |
| `batch_size` | 16 | Taille des batches |

---

## 🔍 Détails Techniques

### **Normalisation des Strokes**
- Les offsets sont normalisés par la médiane de leur norme
- Évite les problèmes d'échelle
- Rend l'entraînement plus stable

### **Padding et Truncation**
- Strokes : Padding avec `[0, 0, 0]` jusqu'à `MAX_STROKE_LEN`
- Textes : Padding avec `0` (caractère nul) jusqu'à `max_text_len`
- Les longueurs réelles sont stockées séparément

### **Loss Function (LSGAN)**
- Utilise MSE au lieu de BCE
- Plus stable pour l'entraînement
- Labels : `1` pour réel, `0` pour faux

### **Data Augmentation**
- Pas d'augmentation explicite dans le code actuel
- Possibilité d'ajouter : rotation, scaling, noise

---

## 🎓 Conclusion

Ce projet implémente un pipeline complet de génération d'écriture manuscrite, de la préparation des données à l'évaluation, avec deux approches complémentaires (GAN et rendu stylisé) et une suite complète de métriques d'évaluation.


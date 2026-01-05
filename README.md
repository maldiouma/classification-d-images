# Classification d’Images Médicales — Transfert d’Apprentissage (ResNet)

## 🎯 Objectif
Développer un pipeline de classification d’images médicales (binaire ou multi-classes) basé sur le transfert d’apprentissage avec **ResNet18 / ResNet50**, incluant :

- Préparation des données
- Entraînement et évaluation
- Explicabilité via **Grad-CAM**

> Usage recherche uniquement — non destiné au diagnostic clinique.

---

## 🧭 Pipeline

1. **Préparation des données**
   Organisation en `train / val / test` (split par patient si applicable).

2. **Prétraitement**
   Redimensionnement, normalisation, augmentations contrôlées.

3. **Modélisation**
   Fine-tuning de ResNet pré-entraîné ImageNet.

4. **Entraînement**
   Gestion du déséquilibre, early stopping, suivi des métriques.

5. **Évaluation**
   Accuracy, F1-Score, ROC-AUC, matrice de confusion.

6. **Explicabilité**
   Visualisation Grad-CAM pour interprétation des prédictions.

---

## ⚙️ Installation

Prérequis : **Python 3.10+**

```bash
pip install -r requirements.txt
```

---

## ▶️ Utilisation

### Entraîner un modèle

```bash
python scripts/train.py --data-root ./data --model resnet18
```

### Lancer une inférence

```bash
python scripts/inference.py \
  --image path/to/image.jpg \
  --checkpoint best_ft.pt
```

---

## 🗂 Structure du projet

```
config.json
results/
scripts/
  train.py
  inference.py
  prepare_data.py
src/
  data.py
  models.py
  training.py
  evaluation.py
  gradcam.py
requirements.txt
notebook.ipynb
```


## ✔️ Bonnes pratiques

* Séparation train / validation / test
* Split au niveau patient lorsque possible
* Gestion du déséquilibre des classes
* Reproductibilité (seeds, versions)
* Validation sur un jeu tenu à part
* Interprétation prudente des cartes Grad-CAM

---

## ⚠️ Avertissement

Ce projet est destiné à la **recherche et à l’apprentissage**.
Il ne doit pas être utilisé pour des décisions cliniques réelles.

## 👤 Auteur

Aldiouma Mbaye — Data & Machine Learning

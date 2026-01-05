# Détection et classification d’images médicales par transfert d’apprentissage

## Objectif
Ce projet vise à classifier des images médicales (binaire ou multi-classes) en utilisant le transfert d’apprentissage avec ResNet18/50, la data augmentation, la calibration et Grad-CAM pour l’explicabilité.

## Pipeline
1. **Téléchargement des données** : Utilisation de la Kaggle API pour récupérer le dataset (ex : HAM10000, Chest X-Ray Pneumonia, Fashion-MNIST).
2. **Prétraitement et augmentation** : Redimensionnement, flips, color jitter, normalisation.
3. **Chargement des données** : Organisation en train/val/test, DataLoader PyTorch.
4. **Modélisation** : ResNet18/50 pré-entraîné ImageNet, fine-tuning des derniers blocs.
5. **Entraînement** : Cross-entropy, Adam, early stopping, pondération des classes.
6. **Évaluation** : ROC-AUC, PR-AUC, F1, matrice de confusion, calibration.
7. **Explicabilité** : Visualisation Grad-CAM.

## Installation
- Python 3.10+
- PyTorch, torchvision, scikit-learn, matplotlib, seaborn, pytorch-grad-cam, kaggle

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pytorch-grad-cam kaggle
```

## Exécution
1. Télécharger le dataset via Kaggle et organiser les dossiers `data/train`, `data/val`, `data/test`.
2. Lancer le script d’entraînement :
    ```bash
    python train.py
    ```
3. Lancer l’inférence et la visualisation :
    ```bash
    python inference.py --image chemin/vers/image.jpg --checkpoint best_ft.pt
    ```

## Bonnes pratiques
- Reproductibilité (seeds, versions, requirements.txt)
- Early stopping pour éviter l’overfitting
- Pondération des classes pour le déséquilibre
- Data augmentation
- Calibration des probabilités
- Explicabilité via Grad-CAM

## Extensions possibles
- Validation croisée K-fold
- Test Time Augmentation (TTA)
- Ensemble methods
- Calibration isotone
- Déploiement (ONNX, FastAPI, Docker)

## Auteur
Aldiouma Mbaye - MSc Data Engineer, Machine Learning
GitHub : afoumalorian-cmd

© 2025 - ECE Paris
│   ├── config.json      # Configuration d'entraînement
│   └── results/         # Visualisations
├── scripts/
│   ├── train.py         # Script d'entraînement
│   ├── inference.py     # Script d'inférence
│   └── prepare_data.py  # Préparation du dataset
├── src/
│   ├── __init__.py
│   ├── data.py          # Chargement et augmentation
│   ├── models.py        # Construction du modèle
│   ├── training.py      # Boucle d'entraînement
│   ├── evaluation.py    # Évaluation et métriques
│   └── gradcam.py       # Grad-CAM explicabilité
├── requirements.txt     # Dépendances Python
├── README.md            # Cette documentation
└── notebook.ipynb       # Notebook Colab exécutable
```

## 🚀 Installation

### 1. Cloner le dépôt
```bash
git clone <repository-url>
cd medical-imaging-classification
```

### 2. Créer un environnement virtuel
```bash
# Python 3.8+
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📖 Utilisation

### Entraînement du modèle

#### Configuration par défaut
```bash
python scripts/train.py \
    --data-root ./data \
    --model resnet18 \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3
```

#### Avec options avancées
```bash
python scripts/train.py \
    --model resnet50 \
    --num-classes 7 \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-3 \
    --fine-tune-epochs 15 \
    --ft-lr 5e-4 \
    --patience 4 \
    --seed 42 \
    --output-dir ./outputs
```

#### Options disponibles
```
--data-root          Racine du dataset (défaut: ./data)
--img-size           Taille des images (défaut: 224)
--batch-size         Taille des batches (défaut: 64)
--num-workers        Workers DataLoader (défaut: 2)
--model              Architecture (resnet18 | resnet50)
--num-classes        Nombre de classes (défaut: 7)
--epochs             Epochs d'entraînement (défaut: 20)
--lr                 Learning rate initial (défaut: 1e-3)
--weight-decay       Weight decay (défaut: 1e-4)
--dropout            Dropout rate (défaut: 0.2)
--patience           Early stopping patience (défaut: 3)
--fine-tune-epochs   Epochs de fine-tuning (défaut: 10)
--ft-lr              LR fine-tuning (défaut: 5e-4)
--device             Device (auto | cpu | cuda)
--seed               Random seed (défaut: 42)
--output-dir         Répertoire de sortie (défaut: ./outputs)
```

### Inférence et évaluation

#### Sur une image unique
```bash
python scripts/inference.py \
    --image path/to/image.jpg \
    --checkpoint ./outputs/checkpoints/best_ft.pt \
    --gradcam
```

#### Sur l'ensemble de test
```bash
python scripts/inference.py \
    --test-dir ./data/test \
    --checkpoint ./outputs/checkpoints/best_ft.pt
```

## 🔬 Méthodologie

### 1. Prétraitement et Augmentation

**Augmentations d'entraînement:**
- RandomResizedCrop(224, scale=(0.7, 1.0))
- RandomHorizontalFlip(p=0.5)
- RandomVerticalFlip(p=0.3)
- ColorJitter(brightness, contrast, saturation, hue)
- RandomRotation(15°)

**Normalisation ImageNet:**
```
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
```

### 2. Transfer Learning

**Stage 1 - Formation de la tête de classification:**
- Modèle ResNet18/50 pré-entraîné sur ImageNet
- Backbone complètement gelé (requires_grad=False)
- Entraînement de la couche FC uniquement
- Loss: CrossEntropyLoss avec poids de classe
- Optimizer: Adam(lr=1e-3)
- Scheduler: CosineAnnealingLR

**Stage 2 - Fine-tuning:**
- Décongélation de layer4 (derniers blocs résiduels)
- Entraînement conjoint du backbone et de la tête
- LR réduite (5e-4) pour éviter la dégradation
- Early stopping avec patience=2

### 3. Gestion du déséquilibre

**Pondérations de classe:**
```python
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes
criterion = CrossEntropyLoss(weight=class_weights)
```

**Stratégies supplémentaires:**
- Data augmentation agressive
- Dropout(0.2)
- Early stopping
- Stratified splitting

### 4. Évaluation

**Métriques calcul:**
- **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
- **ROC-AUC:** Aire sous la courbe ROC
- **PR-AUC:** Aire sous la courbe Precision-Recall
- **Confusion Matrix:** Matrice de confusion complète

**Calibration:**
- Courbe de calibration (Platt)
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)

### 5. Explicabilité - Grad-CAM

**Grad-CAM (Gradient-weighted Class Activation Mapping):**
- Visualise les régions de l'image influencant la prédiction
- Basé sur les gradients de la classe prédite par rapport aux feature maps
- Layer cible: layer4 (derniers blocs résiduels)

```python
from src.gradcam import GradCAMExplainer

explainer = GradCAMExplainer(model, device)
vis, cam = explainer.visualize(image_tensor, target_class)
explainer.plot_gradcam(image_tensor, target_class)
```

## 📊 Résultats attendus

### HAM10000 (7 classes)
| Métrique | ResNet18 | ResNet50 |
|----------|----------|----------|
| Accuracy | ~75-80% | ~80-85% |
| F1-Score | ~0.75 | ~0.81 |
| ROC-AUC (OvR) | ~0.92 | ~0.94 |

*Note: Résultats dépendent de la stratification et du seed aléatoire*

## 🎓 Bonnes pratiques implémentées

### Reproductibilité
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(SEED)
```

### Gestion des versions
- `requirements.txt` avec versions épinglées
- Configuration JSON sauvegardée avec les résultats
- Checkpoints du meilleur modèle

### Documentation
- Docstrings complets (Google style)
- Commentaires expliquant la logique métier
- README détaillé
- Notebook Colab exécutable

### Tests et validation
- Early stopping pour éviter l'overfitting
- Validation sur set distinct
- Test final sur set tenu à l'écart
- Monitoring de métriques multiples

## ⚠️ Limitations et risques d'overfitting

1. **Taille du dataset:** HAM10000 contient ~10k images, potentiellement insuffisant pour certains modèles
2. **Biais géographique/démographique:** Provenance limitée des images
3. **Imbalance:** Certaines classes bien moins représentées
4. **Distribution shift:** Performance en production peut différer
5. **Grad-CAM:** Visualisations peuvent être trompeuses - ne pas en dépendre seul

**Mitigations:**
- Data augmentation agressive
- Stratification train/val/test
- Class weighting
- Early stopping
- Validation croisée (bonus)
- Analyse d'erreurs qualitatives

## 🔐 Considérations éthiques

- ⚠️ **Ne pas utiliser en diagnostique clinique direct** - usage recherche uniquement
- **Consentement/Anonymisation:** Données médicales nécessitent conformité RGPD/HIPAA
- **Biais détection:** Modèles peuvent perpetuer biais existants
- **Explainabilité:** Grad-CAM ne remplace pas l'expertise médicale
- **Audit régulier:** Tester régulièrement sur données de groupes sous-représentés

## 📚 Extensions / Améliorations futures

### Architectures
- [ ] EfficientNet, ViT (Vision Transformer)
- [ ] Ensemble methods (bagging, stacking)
- [ ] Knowledge distillation pour déploiement

### Techniques
- [ ] Test Time Augmentation (TTA)
- [ ] Mixup / Cutmix data augmentation
- [ ] Label smoothing
- [ ] Calibration isotone
- [ ] Focal loss pour déséquilibre extrême

### Évaluation avancée
- [ ] Validation croisée K-fold
- [ ] Courbes d'apprentissage
- [ ] Analyse d'erreurs par classe
- [ ] Robustness tests (corruption, adversarial)

### Déploiement
- [ ] ONNX export
- [ ] FastAPI inference server
- [ ] Docker containerization
- [ ] CI/CD pipeline

## 📖 Références

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Transfer Learning Best Practices](https://cs231n.github.io/transfer-learning/)
- [Grad-CAM](https://arxiv.org/abs/1610.02055)
- [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📝 Licence

MIT License - Voir LICENSE file

## 👨‍💼 Auteur

Aldiouma Mbaye - MSc Data Engineer, Machine Learning
ECE Paris, 2025

---

**Dernière mise à jour:** December 4, 2025

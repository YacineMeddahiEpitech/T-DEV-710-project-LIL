# Chest X-Ray Pneumonia Detection
## T-DEV-710 — Machine Learning & Deep Learning Project

Binary classification of chest X-ray images: **NORMAL** vs **PNEUMONIA**

Dataset: Chest X-Ray Images (Pneumonia) — Kaggle / Guangzhou Women and Children's Medical Center  
Total: 5 856 grayscale JPEG images split into train / val / test

---

## Project structure

```
.
├── zoidberg.ipynb        # Main Jupyter notebook (full pipeline + visualisations)
├── zoidberg.py           # Standalone Python script (same pipeline)
├── generate_data.py      # Batch augmented image generator (data_creation)
├── datasets/
│   ├── train/            # 5216 images (NORMAL / PNEUMONIA)
│   ├── val/              # 16 images (original split)
│   └── test/             # 624 images
├── saved_models/         # Persisted sklearn pipelines + CNN state dict
└── results/              # All generated charts and plots
```

---

## Pipeline overview

| Step | Description |
|------|-------------|
| Data profiling | pandas summary: counts and resolutions per split/class |
| Data cleansing | Detect corrupt / unreadable images before training |
| 3-way split | Train 80% / Val 20% / Test — stratified split |
| Augmentation | Flip, rotation ±15°, affine (zoom + translate), brightness/contrast |
| Normalisation | All pixel values divided by 255 before training |
| PCA | 4096D → 50D (>85% variance retained) |
| t-SNE | 2D non-linear embedding for feature space visualisation |
| ML Pipeline | `StandardScaler → classifier` via `sklearn.pipeline.Pipeline` |
| Cross-Validation | 5-fold StratifiedKFold (prevents data leakage) |
| Hyperparameter tuning | `GridSearchCV` on Random Forest and Logistic Regression |
| Models compared | Logistic Regression, KNN, Random Forest, SVM, CNN |
| Evaluation metric | ROC-AUC (primary) + Accuracy + F1 Score |
| Persistence | `joblib` (sklearn pipelines) + `torch.save` (CNN weights) |

---

## Packages used

| Package | Role | Justification |
|---------|------|---------------|
| numpy | Array maths | Foundation of all numerical ops; no viable alternative |
| pandas | Data profiling | Tabular stats, readable summaries, easy export |
| matplotlib | Scientific plots | Fine-grained control over axes/labels/legends |
| seaborn | Statistical charts | One-call heatmaps with better defaults |
| Pillow | Image I/O & augmentation | Lightweight, pure-Python, all required transforms |
| scikit-learn | ML pipelines, CV, GridSearch | Leakage-safe Pipeline API, tested GridSearchCV |
| torch + torchvision | CNN training | Dynamic graph, MPS (Apple Silicon), Pythonic API |
| joblib | Model persistence | Memory-mapped numpy arrays; ships with sklearn |

---

## Quickstart

```bash
# Run the full ML + DL pipeline
python zoidberg.py

# Generate augmented images (3 copies per original image)
python generate_data.py --copies 3 --size 64

# Open the notebook for full documented analysis
jupyter notebook zoidberg.ipynb
```

---

## Results

All charts saved to `results/`:

| File | Content |
|------|---------|
| `sample_images.png` | Example X-rays per class |
| `class_distribution.png` | Dataset imbalance (bar + pie) |
| `augmentation_examples.png` | Augmentation transforms visualised |
| `pca_analysis.png` | Scree plot + 2D PCA projection |
| `tsne.png` | t-SNE feature space embedding |
| `confusion_matrices.png` | Best ML vs CNN confusion matrices |
| `roc_curves.png` | All models ROC curves on one plot |
| `cv_vs_test.png` | Cross-validation AUC vs test AUC |
| `cnn_training.png` | Training/validation loss and accuracy |

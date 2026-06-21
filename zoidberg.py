"""
Chest X-Ray Pneumonia Detection — ML & Deep Learning Pipeline
=============================================================
Dataset : Chest X-Ray Images (Pneumonia) — Kaggle / Guangzhou Women and Children's Medical Center
Task    : Binary classification NORMAL vs PNEUMONIA

Package justifications
----------------------
numpy      — Fast vectorized array math; foundation of all numerical pipelines.
pandas     — Data profiling and tabular stats; makes class distribution/split
             summaries readable and easy to export.
matplotlib — De facto standard for scientific plots; fine-grained control over
             axes, labels, legends needed for presentation-quality charts.
seaborn    — High-level statistical plots (heatmaps, bar charts) with one call;
             chosen over pure matplotlib for confusion matrices.
Pillow     — Industry-standard image I/O; supports JPEG, PNG, resize, color
             convert, and all augmentation transforms without heavy deps.
scikit-learn — Complete ML toolkit: pipelines, cross-validation, GridSearchCV,
               preprocessing, metrics. Chosen over manual loops for reliability
               and reproducibility guarantees.
PyTorch    — Dynamic computation graph, MPS acceleration on Apple Silicon,
             and torchvision augmentation transforms. Preferred over TensorFlow
             for its Pythonic API and lightweight install.
joblib     — Parallel model serialisation backed by numpy memmap; ships with
             scikit-learn and is the canonical way to persist sklearn objects.
"""

import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# scikit-learn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "./datasets"
MODELS_DIR  = "./saved_models"
RESULTS_DIR = "./results"

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE_ML  = 64   # flat pixel vector for sklearn models
IMG_SIZE_CNN = 64   # spatial input for CNN


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PROFILING & CLEANSING
# ══════════════════════════════════════════════════════════════════════════════

def profile_dataset(data_path: str) -> tuple[pd.DataFrame, list]:
    """Data profiling: per-split/class counts and resolution stats using pandas.
    Data cleansing: detect and report corrupt / unreadable image files.
    """
    records, corrupt = [], []

    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            folder = Path(data_path) / split / cls
            if not folder.exists():
                continue
            files = (list(folder.glob("*.jpeg")) +
                     list(folder.glob("*.jpg")) +
                     list(folder.glob("*.png")))
            widths, heights = [], []
            for f in files:
                try:
                    with Image.open(f) as img:
                        widths.append(img.width)
                        heights.append(img.height)
                except Exception:
                    corrupt.append(str(f))
            if widths:
                records.append({
                    "split":   split,
                    "class":   cls,
                    "count":   len(widths),
                    "mean_w":  round(float(np.mean(widths))),
                    "mean_h":  round(float(np.mean(heights))),
                    "min_res": f"{min(widths)}×{min(heights)}",
                    "max_res": f"{max(widths)}×{max(heights)}",
                })

    df = pd.DataFrame(records)
    print("\n" + "═" * 72)
    print("DATA PROFILING (pandas)")
    print("═" * 72)
    print(df.to_string(index=False))
    total = df["count"].sum()
    print(f"\nTotal images in dataset : {total}")
    print(f"Corrupt / unreadable    : {len(corrupt)}")
    if corrupt:
        for c in corrupt:
            print(f"  ✗ {c}")
    print("═" * 72)
    return df, corrupt


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING — PROPER 3-WAY SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def load_folder(folder: Path, label: int, max_images: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load grayscale images as flat normalised vectors (proc_speed: /255)."""
    if not folder.exists():
        return np.zeros((0, IMG_SIZE_ML * IMG_SIZE_ML)), np.zeros(0, dtype=int)
    files = sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.jpg"))
    if max_images:
        files = files[:max_images]
    imgs, labels = [], []
    for f in files:
        try:
            arr = np.array(
                Image.open(f).convert("L").resize((IMG_SIZE_ML, IMG_SIZE_ML))
            ).flatten() / 255.0          # normalise → speeds up gradient-based models
            imgs.append(arr)
            labels.append(label)
        except Exception:
            continue
    return np.array(imgs), np.array(labels, dtype=int)


def load_dataset(data_path: str, max_per_class_train: int = 350,
                 max_per_class_test: int = 150):
    """Load train/val/test splits.

    The dataset ships with 3 folders (train/val/test).
    The 'val' folder contains only 8+8=16 images — too small for meaningful
    validation metrics.  We therefore carve 20% from the training subset as
    our validation set (stratified), keeping the test folder as held-out data.

    Proportions (with max_per_class_train=350):
        Train  : 560 imgs  (~78%)
        Val    : 140 imgs  (~20%)
        Test   : 300 imgs  (~42% of test folder, used as held-out)
    """
    print("\nLoading images (64×64 grayscale, /255 normalised)…")
    base = Path(data_path)

    tr_n, tr_n_l = load_folder(base / "train" / "NORMAL",    0, max_per_class_train)
    tr_p, tr_p_l = load_folder(base / "train" / "PNEUMONIA", 1, max_per_class_train)
    X_tv = np.vstack([tr_n, tr_p])
    y_tv = np.concatenate([tr_n_l, tr_p_l])

    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.2, stratify=y_tv, random_state=42
    )

    te_n, te_n_l = load_folder(base / "test" / "NORMAL",    0, max_per_class_test)
    te_p, te_p_l = load_folder(base / "test" / "PNEUMONIA", 1, max_per_class_test)
    X_test  = np.vstack([te_n, te_p])
    y_test  = np.concatenate([te_n_l, te_p_l])

    def _fmt(X, y):
        return (f"{len(X)} imgs  "
                f"(N={int((y == 0).sum())}, P={int((y == 1).sum())}, "
                f"{int((y == 1).sum()) / len(y) * 100:.0f}% pneumonia)")

    print(f"  Train : {_fmt(X_train, y_train)}")
    print(f"  Val   : {_fmt(X_val,   y_val)}")
    print(f"  Test  : {_fmt(X_test,  y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test



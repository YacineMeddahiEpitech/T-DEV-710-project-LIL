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


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def visualise_samples(data_path: str):
    """Grid of sample X-ray images with labels (data_visuals)."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Sample Chest X-Rays — NORMAL (top) vs PNEUMONIA (bottom)",
                 fontsize=14, fontweight="bold")

    base = Path(data_path)
    for row, (cls, color) in enumerate([("NORMAL", "green"), ("PNEUMONIA", "red")]):
        files = sorted((base / "train" / cls).glob("*.jpeg"))[:5]
        for col, f in enumerate(files):
            ax = axes[row, col]
            ax.imshow(np.array(Image.open(f).convert("L").resize((128, 128))), cmap="gray")
            ax.set_title(cls, color=color, fontsize=9, fontweight="bold")
            ax.axis("off")
            ax.set_xlabel(f.name[:20], fontsize=6)

    plt.tight_layout()
    p = f"{RESULTS_DIR}/sample_images.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def visualise_class_distribution():
    """Bar + pie charts showing dataset imbalance (data_visuals)."""
    splits = {
        "Train": {"NORMAL": 1341, "PNEUMONIA": 3875},
        "Val":   {"NORMAL": 8,    "PNEUMONIA": 8},
        "Test":  {"NORMAL": 234,  "PNEUMONIA": 390},
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Class Distribution", fontsize=14, fontweight="bold")

    x      = np.arange(len(splits))
    width  = 0.35
    norms  = [v["NORMAL"]    for v in splits.values()]
    pneums = [v["PNEUMONIA"] for v in splits.values()]

    b1 = axes[0].bar(x - width / 2, norms,  width, label="NORMAL",    color="#2196F3", alpha=0.85)
    b2 = axes[0].bar(x + width / 2, pneums, width, label="PNEUMONIA", color="#F44336", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits.keys())
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("Image Count per Split and Class")
    axes[0].legend()
    axes[0].bar_label(b1, padding=3, fontsize=8)
    axes[0].bar_label(b2, padding=3, fontsize=8)

    axes[1].pie(
        [1341, 3875],
        labels=["NORMAL\n(26%)", "PNEUMONIA\n(74%)"],
        colors=["#2196F3", "#F44336"],
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.05, 0],
        shadow=True,
    )
    axes[1].set_title("Train Set Class Imbalance")

    plt.tight_layout()
    p = f"{RESULTS_DIR}/class_distribution.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def visualise_augmentation_examples(data_path: str):
    """Show one original image + 4 augmented variants (data_variation)."""
    from PIL import ImageEnhance
    import random

    src = sorted((Path(data_path) / "train" / "PNEUMONIA").glob("*.jpeg"))[0]
    orig = Image.open(src).convert("L").resize((128, 128))

    def _augment(img, seed):
        r = random.Random(seed)
        img = img.rotate(r.uniform(-20, 20), resample=Image.BILINEAR)
        if r.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = ImageEnhance.Brightness(img).enhance(r.uniform(0.6, 1.4))
        img = ImageEnhance.Contrast(img).enhance(r.uniform(0.7, 1.3))
        scale = r.uniform(0.8, 1.0)
        w, h  = img.size
        nw, nh = int(w * scale), int(h * scale)
        l = r.randint(0, w - nw)
        t = r.randint(0, h - nh)
        img = img.crop((l, t, l + nw, t + nh)).resize((w, h), Image.BILINEAR)
        tx = int(r.uniform(-0.1, 0.1) * w)
        ty = int(r.uniform(-0.1, 0.1) * h)
        img = img.transform(img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty),
                            resample=Image.BILINEAR)
        return img

    variants = [orig] + [_augment(orig, s) for s in range(4)]
    titles   = ["Original", "Rotation+Flip", "Brightness", "Contrast", "Zoom+Shift"]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle("Data Augmentation Examples (PNEUMONIA X-Ray)",
                 fontsize=13, fontweight="bold")
    for ax, img, title in zip(axes, variants, titles):
        ax.imshow(np.array(img), cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    p = f"{RESULTS_DIR}/augmentation_examples.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIMENSIONALITY REDUCTION (PCA + t-SNE)
# ══════════════════════════════════════════════════════════════════════════════

def apply_pca(X_train, X_val, X_test, y_train, n_components: int = 50):
    """Reduce 4096-dim flat vectors with PCA; visualise scree + 2D projection."""
    pca = PCA(n_components=n_components, random_state=42)
    Xtr = pca.fit_transform(X_train)
    Xvl = pca.transform(X_val)
    Xte = pca.transform(X_test)

    var = pca.explained_variance_ratio_.sum() * 100
    print(f"\n  PCA: {X_train.shape[1]}D → {n_components}D  "
          f"(cumulative explained variance: {var:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("PCA — Dimensionality Reduction", fontsize=13, fontweight="bold")

    # Scree / cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[0].plot(cumvar, "b-o", markersize=3)
    axes[0].axhline(95, color="red", linestyle="--", alpha=0.8, label="95% threshold")
    axes[0].axhline(var, color="orange", linestyle="--", alpha=0.8,
                    label=f"{n_components} components ({var:.1f}%)")
    axes[0].set_xlabel("Number of components")
    axes[0].set_ylabel("Cumulative explained variance (%)")
    axes[0].set_title("Scree Plot — Cumulative Variance")
    axes[0].legend(fontsize=8)

    # 2D PCA scatter
    pca2  = PCA(n_components=2, random_state=42)
    X_2d  = pca2.fit_transform(X_train)
    for lbl, color, name in [(0, "#2196F3", "NORMAL"), (1, "#F44336", "PNEUMONIA")]:
        mask = y_train == lbl
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=color, label=name, alpha=0.45, s=14)
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].set_title("2D PCA Projection — Training Set")
    axes[1].legend()

    plt.tight_layout()
    p = f"{RESULTS_DIR}/pca_analysis.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    return Xtr, Xvl, Xte, pca


def visualise_tsne(X_train, y_train, n_samples: int = 250):
    """t-SNE 2D embedding of high-dimensional image vectors."""
    idx  = np.random.default_rng(42).choice(len(X_train), min(n_samples, len(X_train)), replace=False)
    X_s, y_s = X_train[idx], y_train[idx]

    print(f"\n  Computing t-SNE on {len(X_s)} samples…")
    t0   = time.time()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
    X_em = tsne.fit_transform(X_s)
    print(f"  t-SNE done in {time.time() - t0:.1f}s")

    plt.figure(figsize=(8, 6))
    for lbl, color, name in [(0, "#2196F3", "NORMAL"), (1, "#F44336", "PNEUMONIA")]:
        mask = y_s == lbl
        plt.scatter(X_em[mask, 0], X_em[mask, 1], c=color, label=name, alpha=0.6, s=20)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("t-SNE Visualisation of Chest X-Ray Feature Space")
    plt.legend()
    plt.tight_layout()
    p = f"{RESULTS_DIR}/tsne.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ML PIPELINES, CROSS-VALIDATION, TUNING
# ══════════════════════════════════════════════════════════════════════════════

def build_pipelines() -> dict[str, Pipeline]:
    """sklearn Pipeline per model: StandardScaler → classifier.

    Pipelines guarantee that the scaler is fitted only on training folds
    during cross-validation — preventing data leakage.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
    }


def run_cross_validation(pipelines, X_train, y_train, cv_folds: int = 5) -> dict:
    """StratifiedKFold CV — preserves class ratio in every fold."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = {}

    print(f"\n{'─'*65}")
    print(f"  {cv_folds}-Fold Stratified Cross-Validation (scoring: ROC-AUC)")
    print(f"{'─'*65}")

    for name, pipe in pipelines.items():
        t0     = time.time()
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        elapsed = time.time() - t0
        cv_results[name] = scores
        print(f"  {name:<22}  AUC = {scores.mean():.3f} ± {scores.std():.3f}  ({elapsed:.1f}s)")

    return cv_results


def train_and_evaluate(pipelines, X_train, X_val, X_test,
                       y_train, y_val, y_test) -> dict:
    """Fit on train, validate on val, final score on test (proc_tvt)."""
    results = {}
    print(f"\n{'─'*72}")
    print(f"  Train → Val → Test Evaluation")
    print(f"{'─'*72}")
    print(f"  {'Model':<22} {'Val Acc':>8} {'Val AUC':>8} {'Test Acc':>9} {'Test AUC':>9} {'F1':>6}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*9} {'─'*9} {'─'*6}")

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)

        vp  = pipe.predict(X_val);        vpr = pipe.predict_proba(X_val)[:, 1]
        tp  = pipe.predict(X_test);       tpr = pipe.predict_proba(X_test)[:, 1]

        va  = accuracy_score(y_val,  vp);  vau = roc_auc_score(y_val,  vpr)
        ta  = accuracy_score(y_test, tp);  tau = roc_auc_score(y_test, tpr)
        f1  = f1_score(y_test, tp)

        results[name] = dict(
            pipe=pipe, val_pred=vp, val_proba=vpr,
            test_pred=tp, test_proba=tpr,
            val_acc=va, val_auc=vau, test_acc=ta, test_auc=tau, f1=f1,
        )
        print(f"  {name:<22} {va:>8.3f} {vau:>8.3f} {ta:>9.3f} {tau:>9.3f} {f1:>6.3f}")

    return results


def hyperparameter_tuning(X_train, y_train) -> dict:
    """GridSearchCV on Random Forest and Logistic Regression (algo_tuning)."""
    print(f"\n{'─'*65}")
    print("  Hyperparameter Tuning — GridSearchCV (3-fold, scoring: AUC)")
    print(f"{'─'*65}")
    tuned = {}

    rf_pipe = Pipeline([("sc", StandardScaler()),
                        ("clf", RandomForestClassifier(n_jobs=-1, random_state=42))])
    rf_grid = {
        "clf__n_estimators":   [50, 100],
        "clf__max_depth":      [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }
    rf_gs = GridSearchCV(rf_pipe, rf_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    rf_gs.fit(X_train, y_train)
    print(f"  Random Forest  best_params={rf_gs.best_params_}  AUC={rf_gs.best_score_:.3f}")
    tuned["Random Forest"] = rf_gs

    lr_pipe = Pipeline([("sc", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    lr_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    lr_gs = GridSearchCV(lr_pipe, lr_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    lr_gs.fit(X_train, y_train)
    print(f"  Logistic Reg   best_params={lr_gs.best_params_}  AUC={lr_gs.best_score_:.3f}")
    tuned["Logistic Regression"] = lr_gs

    return tuned


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def save_ml_models(results: dict, pca: PCA):
    """Persist all sklearn pipelines + PCA to disk with joblib (data_persistency)."""
    for name, res in results.items():
        slug = name.lower().replace(" ", "_")
        path = f"{MODELS_DIR}/{slug}_pipeline.joblib"
        joblib.dump(res["pipe"], path)
        print(f"  Saved: {path}")
    joblib.dump(pca, f"{MODELS_DIR}/pca.joblib")
    print(f"  Saved: {MODELS_DIR}/pca.joblib")


def load_ml_model(name: str):
    slug = name.lower().replace(" ", "_")
    path = f"{MODELS_DIR}/{slug}_pipeline.joblib"
    return joblib.load(path) if Path(path).exists() else None


def save_cnn_model(model: nn.Module, path: str | None = None):
    p = path or f"{MODELS_DIR}/cnn.pt"
    torch.save(model.state_dict(), p)
    print(f"  Saved CNN state dict: {p}")


def load_cnn_model(path: str | None = None) -> nn.Module | None:
    p = path or f"{MODELS_DIR}/cnn.pt"
    if not Path(p).exists():
        return None
    m = SimpleCNN()
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    m.eval()
    return m



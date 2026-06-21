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


# ══════════════════════════════════════════════════════════════════════════════
# 7. DEEP LEARNING — CNN
# ══════════════════════════════════════════════════════════════════════════════

class XRayDataset(Dataset):
    """PyTorch Dataset with optional augmentation (data_variation, proc_speed)."""

    def __init__(self, data_path: str, split: str,
                 augment: bool = False, max_per_class: int | None = None):
        self.samples = []
        tfms = [transforms.Grayscale(), transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN))]
        if augment:
            tfms += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                # translation ±10%, zoom 0.9–1.1
                transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.90, 1.10)),
                # luminosity variation
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
            ]
        # Normalise to [-1, 1] — speeds up convergence (proc_speed)
        tfms += [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        self.transform = transforms.Compose(tfms)

        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            folder = Path(data_path) / split / cls
            if not folder.exists():
                continue
            files = sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.jpg"))
            if max_per_class:
                files = files[:max_per_class]
            for f in files:
                self.samples.append((str(f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


class SimpleCNN(nn.Module):
    """3-block CNN: Conv-BN-ReLU-MaxPool × 3, then 2 FC layers."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_cnn(data_path: str, epochs: int = 8, batch_size: int = 32,
              lr: float = 1e-3, max_per_class: int = 300):
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  CNN training on {device} — {epochs} epochs, img {IMG_SIZE_CNN}×{IMG_SIZE_CNN}")

    train_ds = XRayDataset(data_path, "train", augment=True,  max_per_class=max_per_class)
    val_ds   = XRayDataset(data_path, "test",  augment=False, max_per_class=max_per_class // 3)
    test_ds  = XRayDataset(data_path, "test",  augment=False, max_per_class=max_per_class // 2)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model     = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    tr_losses, tr_accs, vl_losses, vl_accs = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        run_l, corr, tot = 0.0, 0, 0
        for imgs, lbls in train_ld:
            imgs = imgs.to(device)
            lbls = lbls.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            run_l += loss.item() * imgs.size(0)
            corr  += ((torch.sigmoid(out) > 0.5).long() == lbls.long()).sum().item()
            tot   += imgs.size(0)
        tr_losses.append(run_l / tot)
        tr_accs.append(corr / tot)

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_ld:
                imgs = imgs.to(device)
                lbls = lbls.float().unsqueeze(1).to(device)
                out  = model(imgs)
                vl  += criterion(out, lbls).item() * imgs.size(0)
                vc  += ((torch.sigmoid(out) > 0.5).long() == lbls.long()).sum().item()
                vt  += imgs.size(0)
        vl_losses.append(vl / vt if vt else 0)
        vl_accs.append(vc / vt if vt else 0)

        scheduler.step()
        print(f"  Ep {epoch:>2}/{epochs}  "
              f"train loss={tr_losses[-1]:.4f} acc={tr_accs[-1]:.3f}  "
              f"val loss={vl_losses[-1]:.4f} acc={vl_accs[-1]:.3f}")

    # Final test evaluation
    model.eval()
    preds, probas, gts = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_ld:
            out  = model(imgs.to(device))
            prob = torch.sigmoid(out).squeeze(1).cpu().numpy()
            probas.extend(prob.tolist())
            preds.extend((prob > 0.5).astype(int).tolist())
            gts.extend(lbls.numpy().tolist())

    gts, preds, probas = map(np.array, (gts, preds, probas))
    acc = accuracy_score(gts, preds)
    auc = roc_auc_score(gts, probas)
    print(f"\n  CNN Test — Acc={acc:.3f}  AUC={auc:.3f}  F1={f1_score(gts, preds):.3f}")

    return model, {
        "y_true": gts, "y_pred": preds, "y_proba": probas,
        "acc": acc, "auc": auc,
        "train_losses": tr_losses, "train_accs": tr_accs,
        "val_losses":   vl_losses, "val_accs":   vl_accs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. RESULT VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def visualise_results(ml_results, best_ml, cnn_results, y_test, cv_results):
    """Comprehensive comparison charts (data_visuals)."""

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices — Best ML vs CNN", fontsize=14, fontweight="bold")
    for ax, (name, yt, yp) in zip(axes, [
        (f"Best ML: {best_ml}",     y_test,               ml_results[best_ml]["test_pred"]),
        ("Deep Learning: CNN",       cnn_results["y_true"], cnn_results["y_pred"]),
    ]):
        cm = confusion_matrix(yt, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["NORMAL", "PNEUMONIA"],
                    yticklabels=["NORMAL", "PNEUMONIA"])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curves
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    for i, (name, res) in enumerate(ml_results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["test_proba"])
        plt.plot(fpr, tpr, color=colors[i], linestyle="--",
                 label=f"ML {name} (AUC={res['test_auc']:.3f})")
    fpr, tpr, _ = roc_curve(cnn_results["y_true"], cnn_results["y_proba"])
    plt.plot(fpr, tpr, color="red", linewidth=2.5,
             label=f"CNN (AUC={cnn_results['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — ML Models vs CNN")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # CV comparison + test AUC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison — Cross-Validation vs Test AUC",
                 fontsize=13, fontweight="bold")

    names    = list(cv_results.keys())
    cv_means = [cv_results[n].mean() for n in names]
    cv_stds  = [cv_results[n].std()  for n in names]
    bars = axes[0].barh(names, cv_means, xerr=cv_stds,
                        color="#42A5F5", alpha=0.85, capsize=5)
    axes[0].set_xlabel("Mean ROC-AUC (5-Fold CV)")
    axes[0].set_title("Cross-Validation AUC ± Std (Train Set)")
    axes[0].set_xlim(0.5, 1.0)
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    t_names = list(ml_results.keys())
    t_aucs  = [ml_results[n]["test_auc"] for n in t_names]
    bars2 = axes[1].barh(t_names, t_aucs, color="#EF5350", alpha=0.85)
    axes[1].set_xlabel("Test ROC-AUC Score")
    axes[1].set_title("Test Set AUC Scores")
    axes[1].set_xlim(0.5, 1.0)
    axes[1].bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cv_vs_test.png", dpi=150, bbox_inches="tight")
    plt.close()

    # CNN training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CNN Training Curves", fontsize=13, fontweight="bold")
    eps = range(1, len(cnn_results["train_losses"]) + 1)
    a1.plot(eps, cnn_results["train_losses"], "b-o", ms=4, label="Train Loss")
    a1.plot(eps, cnn_results["val_losses"],   "r-o", ms=4, label="Val Loss")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss")
    a1.set_title("Loss per Epoch"); a1.legend()

    a2.plot(eps, cnn_results["train_accs"], "b-o", ms=4, label="Train Acc")
    a2.plot(eps, cnn_results["val_accs"],   "r-o", ms=4, label="Val Acc")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy")
    a2.set_title("Accuracy per Epoch"); a2.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cnn_training.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  All charts saved to {RESULTS_DIR}/")

    # Final summary table
    print("\n" + "═" * 72)
    print("  FINAL COMPARISON SUMMARY")
    print("═" * 72)
    print(f"  {'Model':<28} {'Test Acc':>9} {'Test AUC':>9} {'F1 Score':>9}")
    print(f"  {'─'*28} {'─'*9} {'─'*9} {'─'*9}")
    for name, res in ml_results.items():
        mark = "  ← best ML" if name == best_ml else ""
        print(f"  ML  {name:<24} {res['test_acc']:>9.3f} {res['test_auc']:>9.3f} "
              f"{res['f1']:>9.3f}{mark}")
    cnn_f1 = f1_score(cnn_results["y_true"], cnn_results["y_pred"])
    print(f"  DL  {'CNN (SimpleCNN)':<24} {cnn_results['acc']:>9.3f} "
          f"{cnn_results['auc']:>9.3f} {cnn_f1:>9.3f}  ← Deep Learning")
    print("═" * 72)

    print(f"\nClassification Report — {best_ml}:")
    print(classification_report(y_test, ml_results[best_ml]["test_pred"],
                                target_names=["NORMAL", "PNEUMONIA"]))
    print("Classification Report — CNN:")
    print(classification_report(cnn_results["y_true"], cnn_results["y_pred"],
                                target_names=["NORMAL", "PNEUMONIA"]))


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 72)
    print("  CHEST X-RAY PNEUMONIA DETECTION — Complete ML & DL Pipeline")
    print("═" * 72)

    # ── Step 1: Data profiling & cleansing ──────────────────────────────────
    print("\n[1/9] Data profiling & cleansing")
    profile_df, corrupt = profile_dataset(DATA_PATH)

    # ── Step 2: Data visualisation ──────────────────────────────────────────
    print("\n[2/9] Data visualisation")
    visualise_samples(DATA_PATH)
    visualise_class_distribution()
    visualise_augmentation_examples(DATA_PATH)

    # ── Step 3: Load with proper 3-way split ────────────────────────────────
    print("\n[3/9] Loading dataset (3-way split)")
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(DATA_PATH)

    # ── Step 4: Dimensionality reduction ────────────────────────────────────
    print("\n[4/9] Dimensionality reduction")
    X_tr_pca, X_vl_pca, X_te_pca, pca = apply_pca(X_train, X_val, X_test, y_train, n_components=50)
    visualise_tsne(X_train, y_train, n_samples=300)

    # ── Step 5: Build sklearn Pipelines ─────────────────────────────────────
    print("\n[5/9] Building sklearn Pipelines")
    pipelines = build_pipelines()
    print(f"  {len(pipelines)} pipelines ready: {list(pipelines.keys())}")

    # ── Step 6: Cross-validation ────────────────────────────────────────────
    print("\n[6/9] Cross-validation")
    cv_results = run_cross_validation(pipelines, X_train, y_train)

    # ── Step 7: Train-Val-Test evaluation ───────────────────────────────────
    print("\n[7/9] Train-Val-Test evaluation")
    ml_results = train_and_evaluate(pipelines, X_train, X_val, X_test,
                                    y_train, y_val, y_test)
    best_ml = max(ml_results, key=lambda n: ml_results[n]["test_auc"])
    print(f"\n  → Best ML model: {best_ml}")

    # ── Step 8: Hyperparameter tuning ───────────────────────────────────────
    print("\n[8/9] Hyperparameter tuning (GridSearchCV)")
    hyperparameter_tuning(X_train, y_train)

    # ── Step 8b: Persist ML models ──────────────────────────────────────────
    print("\n  Saving ML models…")
    save_ml_models(ml_results, pca)

    # ── Step 9: CNN ─────────────────────────────────────────────────────────
    cnn_model_path = f"{MODELS_DIR}/cnn.pt"
    if Path(cnn_model_path).exists():
        print(f"\n[9/9] CNN — loading saved model from {cnn_model_path}")
        cnn_model = load_cnn_model(cnn_model_path)
        print("  Saved model loaded successfully. Re-training skipped.")
        # Quick evaluation on test set
        test_ds = XRayDataset(DATA_PATH, "test", augment=False, max_per_class=150)
        ld = DataLoader(test_ds, batch_size=32, num_workers=0)
        cnn_model.eval()
        pds, prs, gts = [], [], []
        with torch.no_grad():
            for imgs, lbls in ld:
                prob = torch.sigmoid(cnn_model(imgs)).squeeze(1).numpy()
                prs.extend(prob.tolist())
                pds.extend((prob > 0.5).astype(int).tolist())
                gts.extend(lbls.numpy().tolist())
        gts, pds, prs = map(np.array, (gts, pds, prs))
        cnn_results = {
            "y_true": gts, "y_pred": pds, "y_proba": prs,
            "acc": accuracy_score(gts, pds), "auc": roc_auc_score(gts, prs),
            "train_losses": [], "train_accs": [], "val_losses": [], "val_accs": [],
        }
        print(f"  CNN Test — Acc={cnn_results['acc']:.3f}  AUC={cnn_results['auc']:.3f}")
    else:
        print("\n[9/9] CNN — training from scratch")
        cnn_model, cnn_results = train_cnn(DATA_PATH, epochs=8)
        save_cnn_model(cnn_model)

    # ── Final: Visualise & compare ──────────────────────────────────────────
    if cnn_results["train_losses"]:
        visualise_results(ml_results, best_ml, cnn_results, y_test, cv_results)

    print("\n" + "═" * 72)
    print("  Pipeline complete. All results saved to ./results/ and ./saved_models/")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()

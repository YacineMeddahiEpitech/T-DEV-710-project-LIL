"""Quick runtime test — verifies the ML portion of the pipeline runs end-to-end."""
import matplotlib
matplotlib.use("Agg")  # no display needed
import os, warnings
os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)
warnings.filterwarnings("ignore")

from zoidberg import (
    profile_dataset, load_dataset,
    visualise_samples, visualise_class_distribution, visualise_augmentation_examples,
    apply_pca, visualise_tsne,
    build_pipelines, run_cross_validation, train_and_evaluate,
    hyperparameter_tuning, save_ml_models,
)

print("[1] Profiling...")
profile_dataset("./datasets")

print("[2] Visualisations...")
visualise_samples("./datasets")
visualise_class_distribution()
visualise_augmentation_examples("./datasets")

print("[3] Loading data (small subset)...")
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
    "./datasets", max_per_class_train=150, max_per_class_test=75
)

print("[4] PCA...")
X_tr_pca, X_vl_pca, X_te_pca, pca = apply_pca(
    X_train, X_val, X_test, y_train, n_components=30
)

print("[5] t-SNE (100 samples)...")
visualise_tsne(X_train, y_train, n_samples=100)

print("[6] Building pipelines...")
pipelines = build_pipelines()

print("[7] Cross-validation (3-fold, fast)...")
cv = run_cross_validation(pipelines, X_train, y_train, cv_folds=3)

print("[8] Train-Val-Test evaluation...")
ml_results = train_and_evaluate(
    pipelines, X_train, X_val, X_test, y_train, y_val, y_test
)

print("[9] Hyperparameter tuning (small grid)...")
hyperparameter_tuning(X_train, y_train)

print("[10] Saving models...")
save_ml_models(ml_results, pca)

print("\nResults dir:", sorted(os.listdir("results")))
print("Models dir: ", sorted(os.listdir("saved_models")))
print("\n=== QUICK TEST PASSED ===")

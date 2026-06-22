import numpy as np
from pathlib import Path

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import torch

import zoidberg


def _create_image(path: Path, color: int = 128, size: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(size, color, dtype=np.uint8), mode="L").save(path)


def test_load_folder_missing_dir_returns_empty_arrays(tmp_path):
    images, labels = zoidberg.load_folder(tmp_path / "missing", label=1)

    assert images.shape == (0, zoidberg.IMG_SIZE_ML * zoidberg.IMG_SIZE_ML)
    assert labels.shape == (0,)
    assert labels.dtype == int


def test_profile_dataset_counts_valid_images_and_flags_corrupt_files(tmp_path):
    valid_path = tmp_path / "train" / "NORMAL" / "valid.jpeg"
    corrupt_path = tmp_path / "train" / "NORMAL" / "corrupt.jpeg"
    _create_image(valid_path, color=90)
    corrupt_path.write_text("not an image", encoding="utf-8")

    df, corrupt = zoidberg.profile_dataset(str(tmp_path))

    assert len(df) == 1
    row = df.iloc[0]
    assert row["split"] == "train"
    assert row["class"] == "NORMAL"
    assert row["count"] == 1
    assert str(corrupt_path) in corrupt


def test_build_pipelines_returns_expected_models():
    pipelines = zoidberg.build_pipelines()

    assert set(pipelines) == {
        "Logistic Regression",
        "KNN",
        "Random Forest",
        "SVM",
    }
    assert isinstance(pipelines["Logistic Regression"].named_steps["clf"], LogisticRegression)
    assert isinstance(pipelines["KNN"].named_steps["clf"], KNeighborsClassifier)
    assert isinstance(pipelines["Random Forest"].named_steps["clf"], RandomForestClassifier)
    assert isinstance(pipelines["SVM"].named_steps["clf"], SVC)


def test_apply_pca_reduces_all_splits_to_requested_dimensions():
    rng = np.random.default_rng(42)
    x_train = rng.random((12, 64))
    x_val = rng.random((4, 64))
    x_test = rng.random((5, 64))
    y_train = np.array([0, 1] * 6)

    x_train_pca, x_val_pca, x_test_pca, pca = zoidberg.apply_pca(
        x_train, x_val, x_test, y_train, n_components=5
    )

    assert x_train_pca.shape == (12, 5)
    assert x_val_pca.shape == (4, 5)
    assert x_test_pca.shape == (5, 5)
    assert isinstance(pca, PCA)
    assert pca.n_components == 5


def test_save_and_load_ml_model_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(zoidberg, "MODELS_DIR", str(tmp_path))

    pipeline = zoidberg.build_pipelines()["Logistic Regression"]
    pipeline.fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))

    zoidberg.save_ml_models({"Logistic Regression": {"pipe": pipeline}}, PCA(n_components=1))
    loaded = zoidberg.load_ml_model("Logistic Regression")

    assert loaded is not None
    assert loaded.predict(np.array([[0.0], [3.0]])).tolist() == [0, 1]


def test_load_folder_respects_max_images_and_normalizes_pixels(tmp_path):
    folder = tmp_path / "train" / "NORMAL"
    _create_image(folder / "a.jpeg", color=0)
    _create_image(folder / "b.jpeg", color=127)
    _create_image(folder / "c.jpg", color=255)

    images, labels = zoidberg.load_folder(folder, label=0, max_images=2)

    assert images.shape == (2, zoidberg.IMG_SIZE_ML * zoidberg.IMG_SIZE_ML)
    assert labels.tolist() == [0, 0]
    assert np.all(images >= 0.0)
    assert np.all(images <= 1.0)


def test_load_dataset_returns_expected_split_sizes(tmp_path):
    for split, count in [("train", 10), ("test", 4)]:
        for idx in range(count):
            _create_image(tmp_path / split / "NORMAL" / f"n_{idx}.jpeg", color=20 + idx)
            _create_image(tmp_path / split / "PNEUMONIA" / f"p_{idx}.jpeg", color=200 - idx)

    x_train, x_val, x_test, y_train, y_val, y_test = zoidberg.load_dataset(
        str(tmp_path), max_per_class_train=10, max_per_class_test=4
    )

    assert x_train.shape[0] == 16
    assert x_val.shape[0] == 4
    assert x_test.shape[0] == 8
    assert int((y_train == 0).sum()) == 8
    assert int((y_train == 1).sum()) == 8
    assert int((y_val == 0).sum()) == 2
    assert int((y_val == 1).sum()) == 2
    assert int((y_test == 0).sum()) == 4
    assert int((y_test == 1).sum()) == 4


def test_load_ml_model_returns_none_when_model_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(zoidberg, "MODELS_DIR", str(tmp_path))

    assert zoidberg.load_ml_model("Missing Model") is None


def test_xraydataset_returns_tensor_with_expected_shape(tmp_path):
    _create_image(tmp_path / "train" / "NORMAL" / "n1.jpeg", color=10)
    _create_image(tmp_path / "train" / "NORMAL" / "n2.jpeg", color=30)
    _create_image(tmp_path / "train" / "PNEUMONIA" / "p1.jpeg", color=220)

    dataset = zoidberg.XRayDataset(str(tmp_path), "train", augment=False)
    image, label = dataset[0]

    assert len(dataset) == 3
    assert tuple(image.shape) == (1, zoidberg.IMG_SIZE_CNN, zoidberg.IMG_SIZE_CNN)
    assert label in {0, 1}
    assert torch.is_tensor(image)


def test_simplecnn_forward_returns_binary_logit_per_sample():
    model = zoidberg.SimpleCNN()
    batch = torch.randn(2, 1, zoidberg.IMG_SIZE_CNN, zoidberg.IMG_SIZE_CNN)

    output = model(batch)

    assert tuple(output.shape) == (2, 1)

"""Training and persistence helpers for the fitted digital-twin models."""

from datetime import datetime
import hashlib
import os
from pathlib import Path
import platform

import joblib
import numpy
import pandas
import sklearn

from pipeline import TwinDataset, WellDigitalTwin


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "model_artifacts" / "baghewala_models.pkl"
INPUT_FILES = (
    "well_master.csv",
    "pvt_viscosity_samples.csv",
    "css_cycles.csv",
    "daily_production.csv",
    "srp_operations.csv",
    "dynamometer_cards.json",
    "rod_failures.csv",
)
ARTIFACT_SCHEMA_VERSION = 1


def resolve_data_dir(configured=None) -> Path:
    value = configured or os.getenv("BAGHEWALA_DATA_DIR")
    if value:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    generated = ROOT / "baghewala_dataset"
    return generated if generated.is_dir() else ROOT / "baghewala_sample_dataset"


def dataset_fingerprint(data_dir: str | Path) -> str:
    directory = Path(data_dir)
    digest = hashlib.sha256()
    for name in INPUT_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Required model input is missing: {path}")
        digest.update(name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def train_and_save(data_dir: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    data_dir = Path(data_dir).resolve()
    model_path = Path(model_path).resolve()
    dataset = TwinDataset.load(str(data_dir))
    twin = WellDigitalTwin(dataset)
    fingerprint = dataset_fingerprint(data_dir)
    bundle = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training_data_dir": str(data_dir),
        "training_data_fingerprint": fingerprint,
        "training_summary": {
            "well_count": int(dataset.well_master.well_id.nunique()),
            "cycle_count": int(len(dataset.css_cycles)),
            "card_count": int(len(dataset.cards)),
            "rod_failure_count": int(len(dataset.rod_failures)),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "models": {
            "gap1_forecaster": twin.forecaster,
            "gap2_card_classifier": twin.card_classifier,
            "gap3_rod_risk_model": twin.rod_risk_model,
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_name(f"{model_path.stem}.{os.getpid()}.tmp")
    try:
        joblib.dump(bundle, temporary_path, compress=3)
        os.replace(temporary_path, model_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return bundle


def load_model_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model artifact not found: {model_path}. "
            "Train and save it first with `python train_models.py`."
        )
    bundle = joblib.load(model_path)
    if bundle.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported model artifact schema; retrain with train_models.py.")
    required_models = {
        "gap1_forecaster", "gap2_card_classifier", "gap3_rod_risk_model",
    }
    if not required_models.issubset(bundle.get("models", {})):
        raise RuntimeError("Model artifact is incomplete; retrain with train_models.py.")
    trained_version = bundle.get("versions", {}).get("scikit_learn")
    if trained_version != sklearn.__version__:
        raise RuntimeError(
            f"Artifact was trained with scikit-learn {trained_version}, "
            f"but this runtime has {sklearn.__version__}; install the matching version "
            "or retrain the artifact."
        )
    return bundle
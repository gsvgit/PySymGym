import os
from pathlib import Path

TRAINED_MODELS_PATH = Path(os.path.join("report", "trained_models"))
TRAINING_RESULTS_PATH = Path(os.path.join("report", "run_tables"))
PRETRAINED_MODEL_PATH = Path(os.path.join("ml", "models"))
RAW_DATASET_PATH = Path(os.path.join("report", "SerializedEpisodes"))
PROCESSED_DATASET_PATH = Path(os.path.join("report", "dataset"))
LOG_PATH = Path("./ml_app.log")

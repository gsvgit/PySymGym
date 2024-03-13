from ml.data_loader_compact import ServerDataloaderHeteroVector
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2.model import StateModelEncoder
from ml.het_gnn_test_train import HetGNNTestTrain
from ml.common_model.paths import (
    RAW_FILES_PATH,
)
import os
import multiprocessing as mp
from pathlib import Path

DATASET_PATH = Path(os.path.join("report", "pretrain_dataset"))
if not DATASET_PATH.exists():
    os.makedirs(DATASET_PATH)


def get_data_hetero_vector():
    dl = ServerDataloaderHeteroVector(RAW_FILES_PATH, DATASET_PATH)
    dl.save_dataset_for_pretraining(mp.cpu_count() - 1)


if __name__ == "__main__":
    get_data_hetero_vector()
    pr = HetGNNTestTrain(
        StateModelEncoder,
        64,
    )
    pr.train_and_save(DATASET_PATH, 20, "./ml/models/")

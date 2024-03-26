import os
import typing as t
from pathlib import Path

import numpy as np
import torch


def euclidean_dist(y_pred, y_true):
    if len(y_pred) > 1:
        y_pred_min, ind1 = torch.min(y_pred, dim=0)
        y_pred_norm = y_pred - y_pred_min

        y_true_min, ind1 = torch.min(y_true, dim=0)
        y_true_norm = y_true - y_true_min
        return torch.sqrt(torch.sum((y_pred_norm - y_true_norm) ** 2))
    else:
        return 0


def get_model(
    path_to_weights: Path, model_init: t.Callable[[], torch.nn.Module]
) -> torch.nn.Module:
    model = model_init()
    weights = torch.load(path_to_weights)
    weights["lin_last.weight"] = torch.tensor(np.random.random([1, 8]))
    weights["lin_last.bias"] = torch.tensor(np.random.random([1]))
    model.load_state_dict(weights)
    return model


def create_folders_if_necessary(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            os.makedirs(path)


def create_file(file: Path):
    open(file, "w").close()


def append_to_file(file: Path, s: str):
    with open(file, "a") as file:
        file.write(s)

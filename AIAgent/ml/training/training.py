import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import optuna
import torch
import torch.nn as nn
import tqdm
from common.game import GameMap
from config import GeneralConfig

from ml.training.dataset import TrainingDataset
from ml.training.paths import (
    PROCESSED_DATASET_PATH,
    TRAINING_RESULTS_PATH,
    RAW_DATASET_PATH,
    LOG_PATH,
    TRAINED_MODELS_PATH,
)
from ml.training.utils import get_model, create_file
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2.model_modified import (
    StateModelEncoderLastLayer,
)
import optuna
from ml.training.validation import validate
from ml.training.dataloader import DataLoader


@dataclass
class TrialSettings:
    lr: float
    epochs: int
    batch_size: int
    optimizer: torch.optim.Optimizer
    loss: any
    random_seed: int


def train(trial: optuna.trial.Trial, maps: list[GameMap]):
    config = TrialSettings(
        lr=trial.suggest_float("lr", 1e-7, 1e-3),
        batch_size=trial.suggest_int("batch_size", 32, 1024),
        epochs=10,
        optimizer=trial.suggest_categorical("optimizer", [torch.optim.Adam]),
        loss=trial.suggest_categorical("loss", [nn.KLDivLoss]),
        random_seed=937,
    )
    np.random.seed(config.random_seed)
    # path_to_weights = os.path.join(
    #     PRETRAINED_MODEL_PATH,
    #     "RGCNEdgeTypeTAG3VerticesDoubleHistory2",
    #     "64ch",
    #     "20e",
    #     "GNN_state_pred_het_dict",
    # )

    path_to_weights = os.path.join(
        TRAINED_MODELS_PATH,
        "2024-02-04 10:29:16.363173_361_Adam_0.0002933673463885248_KLDL",
        "5",
    )
    model = get_model(
        Path(path_to_weights),
        lambda: StateModelEncoderLastLayer(hidden_channels=64, out_channels=8),
    )
    model.to(GeneralConfig.DEVICE)
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    criterion = config.loss()

    timestamp = datetime.now().timestamp()
    run_name = (
        f"{datetime.fromtimestamp(timestamp)}_{config.batch_size}_Adam_{config.lr}_KLDL"
    )
    print(run_name)
    path_to_trained_models = os.path.join(TRAINED_MODELS_PATH, run_name)
    os.makedirs(path_to_trained_models)
    create_file(LOG_PATH)
    results_table_path = Path(os.path.join(TRAINING_RESULTS_PATH, run_name + ".log"))
    create_file(results_table_path)

    all_average_results = []
    for epoch in range(config.epochs):
        dataset = TrainingDataset(
            RAW_DATASET_PATH,
            PROCESSED_DATASET_PATH,
            maps,
            threshold_coverage=100,
            threshold_steps_number=int(
                np.median(list(map(lambda game_map: game_map.StepsToPlay, maps)))
            ),
        )
        dataloader = DataLoader(dataset, config.batch_size, shuffle=True)
        model.train()
        for batch in tqdm.tqdm(
            dataloader,
            desc="training",
            total=dataloader.number_of_batches,
        ):
            batch.to(GeneralConfig.DEVICE)
            optimizer.zero_grad()
            out = model(
                game_x=batch["game_vertex"].x,
                state_x=batch["state_vertex"].x,
                edge_index_v_v=batch["game_vertex", "to", "game_vertex"].edge_index,
                edge_type_v_v=batch["game_vertex", "to", "game_vertex"].edge_type,
                edge_index_history_v_s=batch[
                    "game_vertex", "history", "state_vertex"
                ].edge_index,
                edge_attr_history_v_s=batch[
                    "game_vertex", "history", "state_vertex"
                ].edge_attr,
                edge_index_in_v_s=batch["game_vertex", "in", "state_vertex"].edge_index,
                edge_index_s_s=batch[
                    "state_vertex", "parent_of", "state_vertex"
                ].edge_index,
            )
            loss = criterion(out, batch.y_true)
            if loss != 0:
                loss.backward()
                optimizer.step()
            del out
            del batch
        val_maps = list(filter(lambda game_map: game_map.StepsToPlay <= 200000, maps))
        average_result = validate(model, val_maps, dataset, epoch, results_table_path)
        all_average_results.append(average_result)
        path_to_model = os.path.join(TRAINED_MODELS_PATH, run_name, str(epoch + 1))
        torch.save(model.state_dict(), Path(path_to_model))

    return max(all_average_results)

import os
import typing as t
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import tqdm
from common.game import GameMap
from config import GeneralConfig
from ml.training.dataloader import DataLoader
from ml.training.dataset import TrainingDataset
from ml.training.paths import (
    LOG_PATH,
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
    TRAINED_MODELS_PATH,
    TRAINING_RESULTS_PATH,
)
from ml.training.utils import create_file
from ml.training.validation import validate


def forward_pass(model, batch):
    return model(
        game_x=batch["game_vertex"].x,
        state_x=batch["state_vertex"].x,
        edge_index_v_v=batch["game_vertex", "to", "game_vertex"].edge_index,
        edge_type_v_v=batch["game_vertex", "to", "game_vertex"].edge_type,
        edge_index_history_v_s=batch[
            "game_vertex", "history", "state_vertex"
        ].edge_index,
        edge_attr_history_v_s=batch["game_vertex", "history", "state_vertex"].edge_attr,
        edge_index_in_v_s=batch["game_vertex", "in", "state_vertex"].edge_index,
        edge_index_s_s=batch["state_vertex", "parent_of", "state_vertex"].edge_index,
    )


@dataclass
class TrialSettings:
    lr: float
    epochs: int
    batch_size: int
    optimizer: torch.optim.Optimizer
    loss: any
    random_seed: int


def train(
    trial: optuna.trial.Trial,
    maps: list[GameMap],
    init_trial_settings: t.Callable[[optuna.trial.Trial], TrialSettings],
    model: torch.nn.Module,
):
    trial_settings = init_trial_settings(trial)
    np.random.seed(trial_settings.random_seed)

    optimizer = trial_settings.optimizer(model.parameters(), lr=trial_settings.lr)
    criterion = trial_settings.loss()

    timestamp = datetime.now().timestamp()
    run_name = f"{datetime.fromtimestamp(timestamp)}_{trial_settings.batch_size}_Adam_{trial_settings.lr}_KLDL"
    print(run_name)
    path_to_trained_models = os.path.join(TRAINED_MODELS_PATH, run_name)
    os.makedirs(path_to_trained_models)
    create_file(LOG_PATH)
    results_table_path = Path(os.path.join(TRAINING_RESULTS_PATH, run_name + ".log"))
    create_file(results_table_path)

    all_average_results = []
    for epoch in range(trial_settings.epochs):
        dataset = TrainingDataset(
            RAW_DATASET_PATH,
            PROCESSED_DATASET_PATH,
            maps,
            threshold_coverage=100,
            threshold_steps_number=int(
                np.median(list(map(lambda game_map: game_map.StepsToPlay, maps)))
            ),
        )
        dataloader = DataLoader(dataset, trial_settings.batch_size, shuffle=True)
        model.train()
        for batch in tqdm.tqdm(
            dataloader,
            desc="training",
            total=dataloader.number_of_batches,
        ):
            batch.to(GeneralConfig.DEVICE)
            optimizer.zero_grad()
            out = forward_pass(model, batch)
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

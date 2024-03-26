import argparse
import configparser
import json
import logging
import multiprocessing as mp
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import joblib
import optuna
import torch
from common.game import GameMap
from config import GeneralConfig
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2.model_modified import (
    StateModelEncoderLastLayer,
)
from ml.training.paths import LOG_PATH, PROCESSED_DATASET_PATH, TRAINING_RESULTS_PATH
from ml.training.training import TrialSettings, train
from ml.training.utils import create_folders_if_necessary, get_model

logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)

create_folders_if_necessary([TRAINING_RESULTS_PATH, PROCESSED_DATASET_PATH])


def main():
    parser = argparse.ArgumentParser(
        description="Train a model using configuration from a .ini file."
    )

    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    dataset_base_path = Path(config["DatasetConfig"]["dataset_base_path"]).resolve()
    dataset_description = Path(config["DatasetConfig"]["dataset_description"]).resolve()
    n_startup_trials = config.getint("OptunaConfig", "n_startup_trials")
    n_trials = config.getint("OptunaConfig", "n_trials")
    num_epochs = config.getint("TrainConfig", "epochs")
    path_to_weights = config.get("TrainConfig", "path_to_weights", fallback=None)
    print(GeneralConfig.DEVICE)

    dbg_dict = {}

    with open(dataset_description, "r") as maps_json:
        maps = GameMap.schema().load(json.loads(maps_json.read()), many=True)
        for map in maps:
            if map.MapName in dbg_dict:
                print(map.MapName)
            dbg_dict[map.MapName] = 0
            fullName = os.path.join(dataset_base_path, map.AssemblyFullName)
            map.AssemblyFullName = fullName

    mp.set_start_method("spawn", force=True)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study = optuna.create_study(sampler=sampler, direction="maximize")

    if path_to_weights is not None:
        model = get_model(
            Path(path_to_weights).resolve(),
            lambda: StateModelEncoderLastLayer(hidden_channels=64, out_channels=8),
        )
    else:
        model = StateModelEncoderLastLayer(hidden_channels=64, out_channels=8)
    model.to(GeneralConfig.DEVICE)

    def init_trial_settings(trial: optuna.trial.Trial):
        return TrialSettings(
            lr=trial.suggest_float("lr", 1e-7, 1e-3),
            batch_size=trial.suggest_int("batch_size", 32, 1024),
            epochs=num_epochs,
            optimizer=trial.suggest_categorical("optimizer", [torch.optim.Adam]),
            loss=trial.suggest_categorical("loss", [torch.nn.KLDivLoss]),
            random_seed=937,
        )

    objective = partial(
        train, maps=maps, init_trial_settings=init_trial_settings, model=model
    )
    study.optimize(objective, n_trials=n_trials)
    joblib.dump(study, f"{datetime.fromtimestamp(datetime.now().timestamp())}.pkl")


if __name__ == "__main__":
    main()

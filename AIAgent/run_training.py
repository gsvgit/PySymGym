import argparse
import json
import logging
import os
from datetime import datetime
from functools import partial
import joblib
import optuna
import multiprocessing as mp
from common.game import GameMap
from config import GeneralConfig
from ml.training.dataset import TrainingDataset
from ml.training.paths import (
    PROCESSED_DATASET_PATH,
    TRAINING_RESULTS_PATH,
    RAW_DATASET_PATH,
    LOG_PATH,
)
from ml.training.utils import create_folders_if_necessary
from ml.training.training import train


logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)

create_folders_if_necessary([TRAINING_RESULTS_PATH, PROCESSED_DATASET_PATH])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasetdescription",
        type=str,
        help="full paths to JSON-file with dataset description",
        required=True,
    )
    parser.add_argument(
        "--datasetbasepath",
        type=str,
        help="path to dir with explored dlls",
        required=True,
    )
    parser.add_argument(
        "--generatedataset",
        type=bool,
        help="set this flag if dataset generation is needed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    dataset_base_path = args.datasetbasepath

    print(GeneralConfig.DEVICE)

    dbg_dict = {}

    with open(args.datasetdescription, "r") as maps_json:
        maps = GameMap.schema().load(json.loads(maps_json.read()), many=True)
        for map in maps:
            if map.MapName in dbg_dict:
                print(map.MapName)
            dbg_dict[map.MapName] = 0
            fullName = os.path.join(dataset_base_path, map.AssemblyFullName)
            map.AssemblyFullName = fullName

    mp.set_start_method("spawn", force=True)
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    objective = partial(train, maps=maps)
    study.optimize(objective, n_trials=100)
    joblib.dump(study, f"{datetime.fromtimestamp(datetime.now().timestamp())}.pkl")


if __name__ == "__main__":
    main()

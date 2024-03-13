import multiprocessing as mp
import os
from pathlib import Path
import numpy as np
import torch
import tqdm
from common.game import GameMap
from config import GeneralConfig
from epochs_statistics.tables import create_pivot_table, table_to_string
from learning.play_game import play_game
from ml.training.dataset import TrainingDataset
from ml.training.utils import append_to_file
from ml.training.wrapper import TrainingModelWrapper


def play_game_task(task):
    maps, dataset, wrapper = task[0], task[1], task[2]
    result = play_game(
        with_predictor=wrapper,
        max_steps=GeneralConfig.MAX_STEPS,
        maps=maps,
        with_dataset=dataset,
    )
    return result


def validate(
    model: torch.nn.Module,
    maps: list[GameMap],
    dataset: TrainingDataset,
    epoch: int,
    results_table_path: Path,
):
    model.eval()
    wrapper = TrainingModelWrapper(model)
    tasks = [
        (
            [map],
            dataset,
            wrapper,
        )
        for map in maps
    ]
    with mp.Pool(GeneralConfig.SERVER_COUNT) as p:
        all_results = []
        for result in tqdm.tqdm(
            p.imap_unordered(play_game_task, tasks, chunksize=1),
            desc="validation",
            total=len(tasks),
        ):
            all_results.extend(result)

    print(
        "Average dataset state result",
        np.average(
            list(
                map(
                    lambda dataset_map_result: dataset_map_result[0],
                    dataset.maps_results.values(),
                )
            )
        ),
    )
    average_result = np.average(
        list(
            map(
                lambda map_result: map_result.game_result.actual_coverage_percent,
                all_results,
            )
        )
    )

    table, _, _ = create_pivot_table(
        {wrapper: sorted(all_results, key=lambda x: x.map.MapName)}
    )
    table = table_to_string(table)
    append_to_file(
        results_table_path,
        f"Epoch#{epoch}" + " Average coverage: " + str(average_result) + "\n",
    )
    append_to_file(results_table_path, table + "\n")

    return average_result

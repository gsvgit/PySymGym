import torch
from pathlib import Path

import os
import numpy as np
import random

import logging
import multiprocessing as mp
from torch_geometric.data import HeteroData, Batch
from typing import Dict, Generator
from ml.dataset import Dataset, Result
import glob
import ast
import shutil
from common.game import GameMap
from config import GeneralConfig


class TrainingDataset(Dataset):
    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        maps: list[GameMap],
        threshold_steps_number: int,
        threshold_coverage: int = 100,
        num_processes: int = mp.cpu_count(),
        similar_steps_save_prob=0,
    ):
        super().__init__(raw_dir, processed_dir, num_processes)
        self.maps_results = self._get_results(maps)
        self.maps = maps
        self.similar_steps_save_prob = similar_steps_save_prob
        self.processed_files_paths = self._processed_files_paths(
            threshold_coverage, threshold_steps_number
        )

    def _get_results(self, maps) -> Dict[str, Result]:
        results = dict()
        for map in maps:
            path_to_map_steps = Path(
                os.path.join(self.processed_dir, map.MapName, "result")
            )
            if path_to_map_steps.exists():
                result_file = open(path_to_map_steps)
                result = ast.literal_eval(result_file.read())
                results[map.MapName] = result
                result_file.close()
        return results

    def _processed_files_paths(
        self, threshold_coverage: int, threshold_steps_number: int
    ) -> list[str]:
        all_files = glob.glob(os.path.join(self.processed_dir, "*", "*"))
        all_files = []
        for map_name in os.listdir(self.processed_dir):
            if map_name in self.maps_results:
                if self.maps_results[map_name][0] >= threshold_coverage:
                    all_steps_paths = [
                        file_path
                        for file_path in glob.glob(
                            os.path.join(self.processed_dir, map_name, "*")
                        )
                        if not file_path.endswith("result")
                    ]
                    if len(all_steps_paths) > threshold_steps_number:
                        all_files.extend(
                            random.sample(all_steps_paths, threshold_steps_number)
                        )
                    else:
                        all_files.extend(all_steps_paths)
        return all_files

    def process(self):
        for map_stat in self.load_dataset():
            map_dir = Path(os.path.join(self.processed_dir, map_stat.MapName))
            if not map_dir.exists():
                os.makedirs(map_dir)
            for step_id, step in enumerate(map_stat.Steps):
                step.Graph.y_true = step.StatesDistribution
                torch.save(
                    step.Graph,
                    os.path.join(map_dir, str(step_id) + ".pt"),
                )
            result_file = open(os.path.join(map_dir, "result"), mode="x")
            result_file.write(str(map_stat.Result))
            result_file.close()

    def filter_map_steps(self, map_steps) -> list[HeteroData]:
        filtered_map_steps = []
        for step in map_steps:
            if step["y_true"].size()[0] != 1 and not step["y_true"].isnan().any():
                max_ind = torch.argmax(step["y_true"])
                step["y_true"] = torch.zeros_like(step["y_true"])
                step["y_true"][max_ind] = 1.0
                filtered_map_steps.append(step)
        return filtered_map_steps

    def remove_similar_steps(self, map_steps):
        filtered_map_steps = []
        for step in map_steps:
            if (
                len(filtered_map_steps) != 0
                and step["y_true"].size() == filtered_map_steps[-1]["y_true"].size()
            ):
                cos_d = 1 - torch.sum(
                    (step["y_true"] / torch.linalg.vector_norm(step["y_true"]))
                    * (
                        filtered_map_steps[-1]["y_true"]
                        / torch.linalg.vector_norm(filtered_map_steps[-1]["y_true"])
                    )
                )
                if (
                    cos_d < 1e-7
                    and step["game_vertex"]["x"].size()[0]
                    == filtered_map_steps[-1]["game_vertex"]["x"].size()[0]
                ):
                    step.use_for_train = np.random.choice(
                        [True, False],
                        p=[
                            self.similar_steps_save_prob,
                            1 - self.similar_steps_save_prob,
                        ],
                    )
                else:
                    step.use_for_train = True
            else:
                step.use_for_train = True
            filtered_map_steps.append(step)
        return filtered_map_steps

    def _get_map_steps(self, map_name) -> list[HeteroData]:
        path_to_map_steps = Path(os.path.join(self.processed_dir, map_name))
        map_steps = []
        all_steps_paths = [
            file_path
            for file_path in glob.glob(os.path.join(path_to_map_steps, "*"))
            if not file_path.endswith("result")
        ]
        for path in all_steps_paths:
            map_steps.append(torch.load(path, map_location=GeneralConfig.DEVICE))
        return map_steps

    def update(
        self,
        map_name: str,
        map_result: Result,
        map_steps: list[HeteroData],
    ):
        filtered_map_steps = self.remove_similar_steps(self.filter_map_steps(map_steps))
        if map_name in self.maps_results.keys():
            if self.maps_results[map_name] == map_result and map_result[0] == 100:
                init_steps_num = (
                    len(os.listdir(os.path.join(self.processed_dir, map_name))) - 1
                )
                num_of_steps_to_merge = len(filtered_map_steps)
                filtered_map_steps = self._merge_steps(filtered_map_steps, map_name)
                new_steps_num = len(filtered_map_steps)
                logging.info(
                    f"Steps on map {map_name} were merged with current steps with result {map_result}. {num_of_steps_to_merge} + {init_steps_num} -> {new_steps_num}. "
                )
                self.maps_results[map_name] = map_result
                self._save_steps(map_name, filtered_map_steps, map_result)
            if self.maps_results[map_name] < map_result:
                logging.info(
                    f"The model with result = {self.maps_results[map_name]} was replaced with the model with "
                    f"result = {map_result} on the map {map_name}"
                )
                self.maps_results[map_name] = map_result
                self._save_steps(map_name, filtered_map_steps, map_result)
        else:
            self.maps_results[map_name] = map_result
            self._save_steps(map_name, filtered_map_steps, map_result)
        del filtered_map_steps
        del map_name
        del map_result

    def _save_steps(self, map_name, steps, map_result):
        path_to_map_steps = Path(os.path.join(self.processed_dir, map_name))
        if path_to_map_steps.exists():
            shutil.rmtree(path_to_map_steps)
        os.makedirs(path_to_map_steps)
        for idx, step in enumerate(steps):
            torch.save(step, os.path.join(path_to_map_steps, str(idx) + ".pt"))
        result_file = open(os.path.join(path_to_map_steps, "result"), mode="x")
        result_file.write(str(map_result))
        result_file.close()

    def _merge_steps(self, steps: list[HeteroData], map_name: str):
        merged_steps = []

        def create_dict(steps_list: list[HeteroData]) -> Dict[int, list[HeteroData]]:
            steps_dict = dict()
            for step in steps_list:
                states_num = step["state_vertex"].x.shape[0]
                game_v_num = step["game_vertex"].x.shape[0]
                if states_num + game_v_num in steps_dict.keys():
                    steps_dict[states_num + game_v_num].append(step)
                else:
                    steps_dict[states_num + game_v_num] = [step]
            return steps_dict

        def flatten_and_sort_hetero_data(step: HeteroData) -> (np.ndarray, np.ndarray):
            game_dtype = [
                (f"g{i}", np.float32) for i in range(step["game_vertex"].x.shape[-1])
            ]
            game_v = np.sort(
                step["game_vertex"].x.cpu().numpy().astype(game_dtype),
                order=list(map(lambda x: x[0], game_dtype)),
            )
            states_dtype = [
                (f"s{i}", np.float32) for i in range(step["state_vertex"].x.shape[-1])
            ]
            states = np.sort(
                step["state_vertex"].x.cpu().numpy().astype(states_dtype),
                order=list(map(lambda x: x[0], states_dtype)),
            )
            return game_v, states

        new_steps = create_dict(steps)
        old_steps = create_dict(self._get_map_steps(map_name))

        for vertex_num in new_steps.keys():
            flattened_old_steps = []
            if vertex_num in old_steps.keys():
                for old_step in old_steps[vertex_num]:
                    flattened_old_steps.append(flatten_and_sort_hetero_data(old_step))
            for new_step in new_steps[vertex_num]:
                new_g_v, new_s_v = flatten_and_sort_hetero_data(new_step)
                should_add = True
                for step_num, (old_g_v, old_s_v) in enumerate(flattened_old_steps):
                    if np.array_equal(new_g_v, old_g_v) and np.array_equal(
                        new_s_v, old_s_v
                    ):
                        y_true_sum = (
                            old_steps[vertex_num][step_num].y_true + new_step.y_true
                        )
                        y_true_sum[y_true_sum != 0] = 1

                        old_steps[vertex_num][step_num].y_true = y_true_sum / torch.sum(
                            y_true_sum
                        )
                        should_add = False
                        break
                if should_add:
                    merged_steps.append(new_step)
        merged_steps.extend(sum(old_steps.values(), []))
        return merged_steps

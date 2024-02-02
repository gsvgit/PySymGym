from collections.abc import Sequence
import torch

import os
import numpy as np
import random

import tqdm
import logging
from ml.common_model.utils import load_dataset_state_dict
import csv
from torch_geometric.data import HeteroData
from typing import TypeAlias, Dict


MapName: TypeAlias = str
GameStatistics: TypeAlias = tuple[int, int, int, int]
GameStepHeteroData: TypeAlias = HeteroData
GameStepsOnMapInfo: TypeAlias = tuple[GameStatistics, Sequence[GameStepHeteroData]]


class FullDataset:
    def __init__(
        self,
        dataset_root_path,
        dataset_map_results_file_name,
        similar_steps_save_prob=0,
    ):
        self.dataset_map_results_file_name = dataset_map_results_file_name
        self.dataset_root_path = dataset_root_path
        self.maps_data: dict[str, GameStepsOnMapInfo] = dict()
        self.similar_steps_save_prob = similar_steps_save_prob

    def load(self):
        maps_results = load_dataset_state_dict(self.dataset_map_results_file_name)
        for file_with_map_steps in tqdm.tqdm(
            os.listdir(self.dataset_root_path), desc="data loading"
        ):
            map_steps = torch.load(
                os.path.join(self.dataset_root_path, file_with_map_steps),
                map_location="cpu",
            )
            map_name = file_with_map_steps[:-3]
            filtered_map_steps = self.filter_map_steps(map_steps)
            filtered_map_steps = self.remove_similar_steps(filtered_map_steps)
            self.maps_data[map_name] = (maps_results[map_name], filtered_map_steps)

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

    def filter_map_steps(self, map_steps):
        filtered_map_steps = []
        for step in map_steps:
            if step["y_true"].size()[0] != 1 and not step["y_true"].isnan().any():
                max_ind = torch.argmax(step["y_true"])
                step["y_true"] = torch.zeros_like(step["y_true"])
                step["y_true"][max_ind] = 1.0
                filtered_map_steps.append(step)
        return filtered_map_steps

    def get_plain_data(
        self, map_result_threshold: int = 100, steps_threshold: int = 2000
    ):
        result = []
        for map_result, map_steps in self.maps_data.values():
            if map_result[0] >= map_result_threshold:
                if len(map_steps) > steps_threshold:
                    selected_steps = random.sample(map_steps, steps_threshold)
                else:
                    selected_steps = map_steps
                for step in selected_steps:
                    if step.use_for_train:
                        result.append(step)
        return result

    def save(self):
        values_for_csv = []
        for map_name in self.maps_data.keys():
            values_for_csv.append(
                {
                    "map_name": map_name,
                    "result": self.maps_data[map_name][0],
                }
            )
            torch.save(
                self.maps_data[map_name][1],
                os.path.join(self.dataset_root_path, map_name + ".pt"),
            )
        with open(self.dataset_map_results_file_name, "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["map_name", "result"])
            writer.writerows(values_for_csv)

    def update(
        self,
        map_name: str,
        map_result: tuple[int, int, int, int],
        map_steps: list[HeteroData],
        move_to_cpu=False,
    ):
        if move_to_cpu:
            for x in map_steps:
                x.to("cpu")
        filtered_map_steps = self.remove_similar_steps(self.filter_map_steps(map_steps))
        if map_name in self.maps_data.keys():
            if self.maps_data[map_name][0] == map_result and map_result[0] == 100:
                init_steps_num = len(self.maps_data[map_name][1])
                self.merge_steps(filtered_map_steps, map_name)
                new_steps_num = len(self.maps_data[map_name][1])
                logging.info(
                    f"Steps on map {map_name} were merged with current steps with result {map_result}. {len(filtered_map_steps)} + {init_steps_num} -> {new_steps_num}. "
                )
            if self.maps_data[map_name][0] < map_result:
                logging.info(
                    f"The model with result = {self.maps_data[map_name][0]} was replaced with the model with "
                    f"result = {map_result} on the map {map_name}"
                )
                self.maps_data[map_name] = (map_result, filtered_map_steps)
        else:
            self.maps_data[map_name] = (map_result, filtered_map_steps)

    def merge_steps(self, steps: list[HeteroData], map_name: str):
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
                step["game_vertex"].x.numpy().astype(game_dtype),
                order=list(map(lambda x: x[0], game_dtype)),
            )
            states_dtype = [
                (f"s{i}", np.float32) for i in range(step["state_vertex"].x.shape[-1])
            ]
            states = np.sort(
                step["state_vertex"].x.numpy().astype(states_dtype),
                order=list(map(lambda x: x[0], states_dtype)),
            )
            return game_v, states

        new_steps = create_dict(steps)
        old_steps = create_dict(self.maps_data[map_name][1])

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
        self.maps_data[map_name] = (self.maps_data[map_name][0], merged_steps)

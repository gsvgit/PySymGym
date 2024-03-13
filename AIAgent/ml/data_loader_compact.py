import json
import os.path
import pickle
from pathlib import Path
from os import walk
from typing import Dict, Tuple, TypeAlias, Generator

import numpy as np
import torch
from torch_geometric.data import HeteroData
from collections.abc import Sequence

from common.game import GameState
from dataclasses import dataclass
import tqdm
import multiprocessing as mp
from functools import partial


NUM_NODE_FEATURES = 6
EXPECTED_FILENAME = "expectedResults.txt"
GAMESTATESUFFIX = "_gameState"
STATESUFFIX = "_statesInfo"
MOVEDSTATESUFFIX = "_movedState"

StateId: TypeAlias = int
StateIndex: TypeAlias = int
StateMap: TypeAlias = Dict[StateId, StateIndex]

VertexId: TypeAlias = int
VertexIndex: TypeAlias = int
VertexMap: TypeAlias = Dict[VertexId, VertexIndex]

MapName: TypeAlias = str
CoveragePercent: TypeAlias = int
TestsNumber: TypeAlias = int
ErrorsNumber: TypeAlias = int
StepsNumber: TypeAlias = int
Result: TypeAlias = tuple[CoveragePercent, TestsNumber, ErrorsNumber, StepsNumber]


@dataclass(slots=True)
class Step:
    Graph: TypeAlias = HeteroData
    StatesDistribution: TypeAlias = torch.tensor
    StatesProperties: TypeAlias = torch.tensor


@dataclass(slots=True)
class MapStatistics:
    MapName: TypeAlias = MapName
    Steps: TypeAlias = list[Step]
    Result: TypeAlias = Result


class ServerDataloaderHeteroVector:
    def __init__(self, raw_files_path: Path, processed_files_path: Path):
        self.graph_types_and_data = {}
        self.dataset = []

        self.raw_files_path = raw_files_path
        self.processed_files_path = processed_files_path

    @staticmethod
    def convert_input_to_tensor(input: GameState) -> Tuple[HeteroData, Dict[int, int]]:
        """
        Converts game env to tensors
        """
        graphVertices = input.GraphVertices
        game_states = input.States
        game_edges = input.Map
        data = HeteroData()
        nodes_vertex_set = set()
        nodes_state_set = set()
        nodes_vertex = []
        nodes_state = []
        edges_index_v_v = []
        edges_index_s_s = []
        edges_index_s_v_in = []
        edges_index_v_s_in = []
        edges_index_s_v_history = []
        edges_index_v_s_history = []
        edges_attr_v_v = []
        edges_types_v_v = []

        edges_attr_s_v = []
        edges_attr_v_s = []

        state_map: Dict[int, int] = {}  # Maps real state id to its index
        vertex_map: Dict[int, int] = {}  # Maps real vertex id to its index
        vertex_index = 0
        state_index = 0

        # vertex nodes
        for v in graphVertices:
            vertex_id = v.Id
            if vertex_id not in vertex_map:
                vertex_map[vertex_id] = vertex_index  # maintain order in tensors
                vertex_index = vertex_index + 1
                nodes_vertex.append(
                    np.array(
                        [
                            int(v.InCoverageZone),
                            v.BasicBlockSize,
                            int(v.CoveredByTest),
                            int(v.VisitedByState),
                            int(v.TouchedByState),
                            int(v.ContainsCall),
                            int(v.ContainsThrow),
                        ]
                    )
                )
        # vertex -> vertex edges
        for e in game_edges:
            edges_index_v_v.append(
                np.array([vertex_map[e.VertexFrom], vertex_map[e.VertexTo]])
            )
            edges_attr_v_v.append(np.array([e.Label.Token]))
            edges_types_v_v.append(e.Label.Token)

        state_doubles = 0

        # state nodes
        for s in game_states:
            state_id = s.Id
            if state_id not in state_map:
                state_map[state_id] = state_index
                nodes_state.append(
                    np.array(
                        [
                            s.Position,
                            s.PathConditionSize,
                            s.VisitedAgainVertices,
                            s.VisitedNotCoveredVerticesInZone,
                            s.VisitedNotCoveredVerticesOutOfZone,
                            s.StepWhenMovedLastTime,
                            s.InstructionsVisitedInCurrentBlock,
                        ]
                    )
                )
                # history edges: state -> vertex and back
                for h in s.History:
                    v_to = vertex_map[h.GraphVertexId]
                    edges_index_s_v_history.append(np.array([state_index, v_to]))
                    edges_index_v_s_history.append(np.array([v_to, state_index]))
                    edges_attr_s_v.append(
                        np.array([h.NumOfVisits, h.StepWhenVisitedLastTime])
                    )
                    edges_attr_v_s.append(
                        np.array([h.NumOfVisits, h.StepWhenVisitedLastTime])
                    )
                state_index = state_index + 1
            else:
                state_doubles += 1

        # state and its childen edges: state -> state
        for s in game_states:
            for ch in s.Children:
                try:
                    edges_index_s_s.append(np.array([state_map[s.Id], state_map[ch]]))
                except KeyError:
                    print("[ERROR]: KeyError")
                    return None, None

        # state position edges: vertex -> state and back
        for v in graphVertices:
            for s in v.States:
                edges_index_s_v_in.append(np.array([state_map[s], vertex_map[v.Id]]))
                edges_index_v_s_in.append(np.array([vertex_map[v.Id], state_map[s]]))

        data["game_vertex"].x = torch.tensor(np.array(nodes_vertex), dtype=torch.float)
        data["state_vertex"].x = torch.tensor(np.array(nodes_state), dtype=torch.float)

        def tensor_not_empty(tensor):
            return tensor.numel() != 0

        # dumb fix
        def null_if_empty(tensor):
            return (
                tensor
                if tensor_not_empty(tensor)
                else torch.empty((2, 0), dtype=torch.int64)
            )

        data["game_vertex", "to", "game_vertex"].edge_index = null_if_empty(
            torch.tensor(np.array(edges_index_v_v), dtype=torch.long).t().contiguous()
        )

        data["game_vertex", "to", "game_vertex"].edge_attr = torch.tensor(
            np.array(edges_attr_v_v), dtype=torch.long
        )
        data["game_vertex", "to", "game_vertex"].edge_type = torch.tensor(
            np.array(edges_types_v_v), dtype=torch.long
        )
        data["state_vertex", "in", "game_vertex"].edge_index = (
            torch.tensor(np.array(edges_index_s_v_in), dtype=torch.long)
            .t()
            .contiguous()
        )
        data["game_vertex", "in", "state_vertex"].edge_index = (
            torch.tensor(np.array(edges_index_v_s_in), dtype=torch.long)
            .t()
            .contiguous()
        )

        data["state_vertex", "history", "game_vertex"].edge_index = null_if_empty(
            torch.tensor(np.array(edges_index_s_v_history), dtype=torch.long)
            .t()
            .contiguous()
        )
        data["game_vertex", "history", "state_vertex"].edge_index = null_if_empty(
            torch.tensor(np.array(edges_index_v_s_history), dtype=torch.long)
            .t()
            .contiguous()
        )
        data["state_vertex", "history", "game_vertex"].edge_attr = torch.tensor(
            np.array(edges_attr_s_v), dtype=torch.long
        )
        data["game_vertex", "history", "state_vertex"].edge_attr = torch.tensor(
            np.array(edges_attr_v_s), dtype=torch.long
        )
        # if (edges_index_s_s): #TODO: empty?
        data["state_vertex", "parent_of", "state_vertex"].edge_index = null_if_empty(
            torch.tensor(np.array(edges_index_s_s), dtype=torch.long).t().contiguous()
        )
        return data, state_map

    def process_step(self, map_path: Path, step_id: str) -> Step:
        def get_states_properties(
            file_path: str, state_map: StateMap
        ) -> Step.StatesProperties:
            """Get tensor for states"""
            expected = dict()
            f = open(
                file_path
            )  # without resource manager in order to escape file descriptors leaks
            data = json.load(f)
            state_set = set()
            for d in data:
                sid = d["StateId"]
                if sid not in state_set:
                    state_set.add(sid)
                    values = [
                        d["NextInstructionIsUncoveredInZone"],
                        d["ChildNumberNormalized"],
                        d["VisitedVerticesInZoneNormalized"],
                        d["Productivity"],
                        d["DistanceToReturnNormalized"],
                        d["DistanceToUncoveredNormalized"],
                        d["DistanceToNotVisitedNormalized"],
                        d["ExpectedWeight"],
                    ]
                    expected[sid] = np.array(values)
            f.close()
            ordered = []
            ordered_by_index = list(
                zip(*sorted(state_map.items(), key=lambda x: x[1]))
            )[0]
            for k in ordered_by_index:
                ordered.append(expected[k])
            return torch.tensor(np.array(ordered), dtype=torch.float)

        def get_states_distribution(
            file_path: str, state_map: StateMap
        ) -> Step.StatesDistribution:
            states_distribution = torch.zeros([len(state_map.keys()), 1])
            f = open(file_path)
            state_id = int(f.read())
            f.close()
            states_distribution[state_map[state_id]] = 1
            return states_distribution

        f = open(os.path.join(map_path, step_id + GAMESTATESUFFIX))
        data = json.load(f)
        graph, state_map = self.convert_input_to_tensor(GameState.from_dict(data))
        if graph is not None:
            # add_expected values
            states_properties = get_states_properties(
                os.path.join(map_path, step_id + STATESUFFIX),
                state_map,
            )
        f.close()
        distribution = get_states_distribution(
            os.path.join(map_path, step_id + MOVEDSTATESUFFIX), state_map
        )
        step = Step(graph, distribution, states_properties)
        return step

    def load_dataset(self, num_processes) -> Generator[MapStatistics, None, None]:
        def get_result(map_name: MapName):
            f = open(os.path.join(self.raw_files_path, map_name, "result"))
            result = f.read()
            f.close()
            result = tuple(map(lambda x: int(x), result.split()))
            return (result[0], -result[1], -result[2], result[3])

        def get_step_ids(map_name: str):
            file_names = os.listdir(os.path.join(self.raw_files_path, map_name))
            step_ids = list(
                set(map(lambda file_name: file_name.split("_")[0], file_names))
            )
            step_ids.remove("result")
            return step_ids

        with mp.Pool(num_processes) as p:
            for map_name in tqdm.tqdm(
                os.listdir(self.raw_files_path), desc="Dataset processing"
            ):
                map_path = os.path.join(self.raw_files_path, map_name)
                process_steps_task = partial(self.process_step, map_path)
                ids = get_step_ids(map_name)
                steps = list(p.imap(process_steps_task, sorted(ids), 10))
                yield MapStatistics(
                    MapName=map_name,
                    Steps=steps,
                    Result=get_result(map_name),
                )

    def save_dataset_for_pretraining(self, num_processes=1):
        for map_stat in self.load_dataset(num_processes):
            steps = []
            for step in map_stat.Steps:
                step.Graph.y = step.StatesProperties
                steps.append(step.Graph)
            PIK = os.path.join(self.processed_files_path, map_stat.MapName + ".dat")
            f = open(PIK, "wb")
            pickle.dump(steps, f)
            f.close()

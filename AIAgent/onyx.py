import argparse
import json
import os
import pathlib
import typing as t

import onnx
import onnxruntime
import torch
from torch_geometric.data import HeteroData

from common.game import GameState
from ml.data_loader_compact import ServerDataloaderHeteroVector
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2.model_modified import (
    StateModelEncoderLastLayer,
)


class TORCH:
    game_vertex = "game_vertex"
    state_vertex = "state_vertex"
    gamevertex_to_gamevertex = (
        "game_vertex",
        "to",
        "game_vertex",
    )
    gamevertex_history_statevertex = (
        "game_vertex",
        "history",
        "state_vertex",
    )
    gamevertex_in_statevertex = (
        "game_vertex",
        "in",
        "state_vertex",
    )
    statevertex_parentof_statevertex = (
        "state_vertex",
        "parent_of",
        "state_vertex",
    )


class ONNX:
    game_vertex = "game_vertex"
    state_vertex = "state_vertex"
    gamevertex_to_gamevertex_index = "gamevertex_to_gamevertex_index"
    gamevertex_to_gamevertex_type = "gamevertex_to_gamevertex_type"
    gamevertex_history_statevertex_index = "gamevertex_history_statevertex_index"
    gamevertex_history_statevertex_attrs = "gamevertex_history_statevertex_attrs"
    gamevertex_in_statevertex = "gamevertex_in_statevertex"
    statevertex_parentof_statevertex = "statevertex_parentof_statevertex"


def load_gamestate(f: t.TextIO) -> HeteroData:
    # load game state + convert in to HeteroData
    file_json = GameState.from_dict(json.load(f))
    hetero_data, _ = ServerDataloaderHeteroVector.convert_input_to_tensor(file_json)

    return hetero_data


def create_model_input(
    hetero_data: HeteroData, modifier: t.Callable[[t.Any], t.Any] = lambda x: x
):
    return {
        ONNX.game_vertex: modifier(hetero_data[TORCH.game_vertex].x),
        ONNX.state_vertex: modifier(hetero_data[TORCH.state_vertex].x),
        ONNX.gamevertex_to_gamevertex_index: modifier(
            hetero_data[TORCH.gamevertex_to_gamevertex].edge_index
        ),
        ONNX.gamevertex_to_gamevertex_type: modifier(
            hetero_data[TORCH.gamevertex_to_gamevertex].edge_type
        ),
        ONNX.gamevertex_history_statevertex_index: modifier(
            hetero_data[TORCH.gamevertex_history_statevertex].edge_index
        ),
        ONNX.gamevertex_history_statevertex_attrs: modifier(
            hetero_data[TORCH.gamevertex_history_statevertex].edge_attr
        ),
        ONNX.gamevertex_in_statevertex: modifier(
            hetero_data[TORCH.gamevertex_in_statevertex].edge_index
        ),
        ONNX.statevertex_parentof_statevertex: modifier(
            hetero_data[TORCH.statevertex_parentof_statevertex].edge_index
        ),
    }


create_onnxruntime_input = lambda heterodata: create_model_input(
    heterodata, lambda x: x.numpy()
)


create_torch_input = lambda heterodata: tuple(create_model_input(heterodata).values())


def save_in_onnx(
    model: torch.nn.Module,
    sample_input: pathlib.Path,
    save_path: pathlib.Path,
    verbose: bool = False,
):
    with open(sample_input, "r") as gamestate_file:
        gamestate = load_gamestate(gamestate_file)
    torch.onnx.export(
        model=model,
        args=create_torch_input(gamestate),
        f=save_path,
        verbose=verbose,
        dynamic_axes={
            ONNX.game_vertex: [0],
            ONNX.state_vertex: [0],
            ONNX.gamevertex_to_gamevertex_index: [1],
            ONNX.gamevertex_to_gamevertex_type: [0],
            ONNX.gamevertex_history_statevertex_index: [1],
            ONNX.gamevertex_history_statevertex_attrs: [0, 1],
            ONNX.gamevertex_in_statevertex: [1],
            ONNX.statevertex_parentof_statevertex: [1],
        },
        input_names=[
            ONNX.game_vertex,
            ONNX.state_vertex,
            ONNX.gamevertex_to_gamevertex_index,
            ONNX.gamevertex_to_gamevertex_type,
            ONNX.gamevertex_history_statevertex_index,
            ONNX.gamevertex_history_statevertex_attrs,
            ONNX.gamevertex_in_statevertex,
            ONNX.statevertex_parentof_statevertex,
        ],
        output_names=["out"],
        opset_version=17,
    )


def torch_run(torch_model: torch.nn.Module, data: HeteroData) -> ...:
    torch_in = tuple(create_model_input(data).values())
    return torch_model(*torch_in)


def onnx_run(ort_session: onnxruntime.InferenceSession, data: HeteroData) -> ...:
    return ort_session.run(None, create_onnxruntime_input(data))


def shorten_output(torch_out, round_to: int = 3):
    shortened = list(map(lambda x: round(x, round_to), torch_out.flatten().tolist()))
    if len(shortened) > 10:
        shortened = shortened[:10] + ["..."]
    return shortened


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-gamestate",
        dest="sample_gamestate_path",
        type=pathlib.Path,
        required=True,
        help="Path to game state that will initialize ONNX model",
    )
    parser.add_argument(
        "--pytorch-model",
        dest="pytorch_model_path",
        type=pathlib.Path,
        required=True,
        help="Path to torch model weights",
    )
    parser.add_argument(
        "--savepath",
        dest="onnx_savepath",
        type=pathlib.Path,
        required=True,
        help="Path to which ONNX model will be saved",
    )
    parser.add_argument(
        "--verify-on",
        dest="verification_gamestates",
        type=pathlib.Path,
        nargs="*",
        required=False,
        default=[],
        help="Paths to game states to verify against",
    )

    args = parser.parse_args()

    entrypoint(
        args.sample_gamestate_path,
        args.pytorch_model_path,
        args.onnx_savepath,
        args.verification_gamestates,
    )


def entrypoint(
    sample_gamestate_path: pathlib.Path,
    pytorch_model_path: pathlib.Path,
    onnx_savepath: pathlib.Path,
    verification_gamestates: list[pathlib.Path] = None,
):
    torch_model = StateModelEncoderLastLayer(64, 8)

    state_dict: t.OrderedDict = torch.load(pytorch_model_path, map_location="cpu")

    torch_model.load_state_dict(state_dict)

    save_in_onnx(torch_model, sample_gamestate_path, onnx_savepath)
    model_onnx = onnx.load(onnx_savepath)
    onnx.checker.check_model(model_onnx)

    if verification_gamestates is not []:
        ort_session = onnxruntime.InferenceSession(onnx_savepath)

        for idx, gamestate_path in enumerate(verification_gamestates):
            with open(gamestate_path, "r") as gamestate_file:
                gamestate = load_gamestate(gamestate_file)

            torch_out = torch_run(torch_model, gamestate)
            onnx_out = onnx_run(ort_session, gamestate)

            print(f"{shorten_output(torch_out)=}")
            print(f"{shorten_output(onnx_out[0])=}")
            print(f"{idx}/{len(verification_gamestates)}")
            assert shorten_output(torch_out) == shorten_output(onnx_out[0])


if __name__ == "__main__":
    main()

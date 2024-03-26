from pathlib import Path

import torch

from config import GeneralConfig


def save_model(model: torch.nn.Module, to: Path):
    torch.save(model.state_dict(), to)


def load_model_from_file(model: torch.nn.Module, file: Path):
    model.load_state_dict(torch.load(file))
    model.to(GeneralConfig.DEVICE)
    model.eval()
    return model

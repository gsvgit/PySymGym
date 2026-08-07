from ml.pc_remover import remove_path_condition_root
from pathlib import Path
import os
import torch
import shutil

old_path = Path("/report/dataset_path_condition_copy_03.03.26")
new_path = Path("/report/dataset_path_condition_without_root")

paths = list(old_path.rglob("*"))

os.makedirs(new_path, exist_ok=True)

for path in paths:
    tail = path.relative_to(old_path)
    new_item_path = new_path / tail

    if path.is_dir():
        os.makedirs(new_item_path, exist_ok=True)

    if path.is_file():
        if path.suffix == ".pt":
            hetero_data = torch.load(path, weights_only=False)
            new_heterodata = remove_path_condition_root(hetero_data)
            torch.save(new_heterodata, new_item_path)
        else:
            shutil.copy2(path, new_item_path)

    print(new_item_path)

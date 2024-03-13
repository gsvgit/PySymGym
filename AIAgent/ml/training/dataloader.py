import torch
from torch_geometric.data import Batch
from ml.dataset import Dataset
from config import GeneralConfig
from ml.dataset import Dataset


class DataLoader:
    def __init__(
        self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_batch_num = -1
        self.number_of_batches = len(
            dataset.processed_files_paths
        ) // self.batch_size + (
            not len(dataset.processed_files_paths) % self.batch_size == 0
        )
        if shuffle:
            self.permutation = torch.randperm(len(dataset.processed_files_paths))
        else:
            self.permutation = torch.tensor(range(len(dataset.processed_files_paths)))

    def __iter__(self):
        return self

    def __next__(self):
        def get_batch(permutation: torch.tensor) -> Batch:
            steps = []
            for idx in permutation:
                steps.append(
                    torch.load(
                        self.dataset.processed_files_paths[idx],
                        map_location=GeneralConfig.DEVICE,
                    )
                )
            batch = Batch.from_data_list(steps, exclude_keys=["use_for_train"])
            return batch

        self.current_batch_num += 1
        if self.current_batch_num < self.number_of_batches:
            if self.current_batch_num == self.number_of_batches - 1:
                return get_batch(
                    self.permutation[self.current_batch_num * self.batch_size :]
                )
            else:
                return get_batch(
                    self.permutation[
                        self.current_batch_num
                        * self.batch_size : (self.current_batch_num + 1)
                        * self.batch_size
                    ]
                )
        else:
            raise StopIteration

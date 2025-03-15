from .modelnet40 import ModelNet40
from .scanobjectnn import ScanObjectNN
from .objaverse import ObjaverseLVIS

dataset_list = {
    "modelnet40": ModelNet40,
    "scanobjectnn": ScanObjectNN,
    "objaverse": ObjaverseLVIS,
}


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)

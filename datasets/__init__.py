from .modelnet40 import ModelNet40

dataset_list = {
    "modelnet40": ModelNet40,
}


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)

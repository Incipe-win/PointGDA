import os
import math
import random
import h5py
import numpy as np
from collections import defaultdict
from typing import List, Dict
import torch
from torch.utils.data import Dataset

from datasets.utils import Datum, DatasetBase, read_json, write_json, build_data_loader

template = ["point cloud depth map of a {}."]


class ScanObjectNN(DatasetBase):
    dataset_dir = "scanobjectnn"

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # Load class names
        self._load_class_names()

        self.template = template

        # Read and process dataset splits
        train, val, test = self.read_data()
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    def _load_class_names(self):
        """Load ScanObjectNN class names from shape names file"""
        class_names_path = os.path.join(self.dataset_dir, "shape_names.txt")
        with open(class_names_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def _read_hdf5_data(self, file_path: str) -> List[Datum]:
        """Read HDF5 file and return list of Datum objects"""
        data = []
        with h5py.File(file_path, "r") as f:
            points = np.array(f["data"])
            labels = np.array(f["label"]).squeeze().astype(int)

            for i in range(len(points)):
                # Convert points to float32 tensor and normalize
                point_cloud = torch.from_numpy(points[i]).float()
                point_cloud = self._normalize_pointcloud(point_cloud)

                label = labels[i]
                classname = self.class_names[label]

                data.append(Datum(impath=point_cloud, label=label, classname=classname))
        return data

    def _normalize_pointcloud(self, pointcloud):
        """Normalize point cloud to [-1,1] range (复用ModelNet40的归一化方法)"""
        centroid = torch.mean(pointcloud, dim=0)
        pointcloud -= centroid
        max_dist = torch.max(torch.sqrt(torch.sum(pointcloud**2, dim=1)))
        pointcloud /= max_dist
        return pointcloud

    def read_data(self):
        """Read official train/test split"""
        train_file = os.path.join(self.dataset_dir, "training_objectdataset.h5")
        test_file = os.path.join(self.dataset_dir, "test_objectdataset.h5")

        # Read training data
        train_data = self._read_hdf5_data(train_file)

        # Read test data
        test_data = self._read_hdf5_data(test_file)

        # Split train into train/val
        train, val = self.split_trainval(train_data, p_val=0.2)

        return train, val, test_data

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from datasets import build_dataset
    import yaml
    from tqdm import tqdm

    cfg = yaml.load(
        open("/workspace/code/deep_learning/PointGDA/configs/scanobjectnn.yaml", "r"),
        Loader=yaml.Loader,
    )
    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
    train_loader = build_data_loader(dataset.train_x, batch_size=1, is_train=True)
    print(train_loader)
    print(len(train_loader))

    for _, (pc, target) in enumerate(tqdm(train_loader)):
        points, target = pc.cuda(), target.cuda()
        print(points.shape)
        break

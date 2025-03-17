import os
import random
import numpy as np
from collections import defaultdict
import torch
import math

from datasets.utils import Datum, DatasetBase, build_data_loader

template = [
    "point cloud depth map of a {}.",
    "There is a {} in the scene.",
    "There is the {} in the scene.",
]


class ObjaverseLVIS(DatasetBase):
    dataset_dir = "objaverse_lvis"

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "lvis_testset.txt")

        with open(self.split_path, "r") as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            self.file_list.append(
                {
                    "cate_id": line.split(",")[0],
                    "cate_name": line.split(",")[1],
                    "model_id": line.split(",")[2],
                    "point_path": self.dataset_dir + line.split(",")[3].replace("\n", ""),
                }
            )

        self.template = template

        # Read and process dataset splits
        train, val, test = self.read_data()
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    def _normalize_pointcloud(self, pointcloud):
        """Normalize point cloud to [-1,1] range"""
        centroid = torch.mean(pointcloud, dim=0)
        pointcloud -= centroid
        max_dist = torch.max(torch.sqrt(torch.sum(pointcloud**2, dim=1)))
        pointcloud /= max_dist
        return pointcloud

    def read_data(self):
        data = []
        for idx in range(len(self.file_list)):
            sample = self.file_list[idx]
            cate_id, cate_name, model_id, point_path = (
                sample["cate_id"],
                sample["cate_name"],
                sample["model_id"],
                sample["point_path"],
            )

            openshape_data = np.load(point_path, allow_pickle=True).item()
            pc_data = openshape_data["xyz"].astype(np.float32)
            pc_data = torch.from_numpy(pc_data).float()
            rgb = openshape_data["rgb"].astype(np.float32)
            rgb = torch.from_numpy(rgb).float()
            pc_data = self._normalize_pointcloud(pc_data)
            cate_id = np.array(cate_id).squeeze().astype(np.int64).item()
            data.append(Datum(impath=pc_data, label=cate_id, classname=cate_name, rgb=rgb))

        train, test = self.split_trainval(data, p_val=0.8)
        # train = data
        # test = data
        val = test

        return train, val, test

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
            n_val = math.floor(len(idxs) * p_val)
            # 确保至少有一个样本留在训练集
            if len(idxs) - n_val < 1:
                n_val = len(idxs) - 1
            random.shuffle(idxs)
            val_indices = idxs[:n_val]
            train_indices = idxs[n_val:]
            val.extend([trainval[i] for i in val_indices])
            train.extend([trainval[i] for i in train_indices])

        return train, val


if __name__ == "__main__":
    from datasets import build_dataset
    import yaml
    from tqdm import tqdm
    import torch.nn.functional as F

    cfg = yaml.load(
        open("/workspace/code/deep_learning/PointGDA/configs/objaverse.yaml", "r"),
        Loader=yaml.Loader,
    )
    cfg["shots"] = 16
    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
    train_loader = build_data_loader(dataset.train_x, batch_size=1, is_train=True)
    test_loader = build_data_loader(dataset.test, batch_size=1, is_train=True)
    print(train_loader)
    print(len(train_loader), len(test_loader))

    for _, (pc, target, rgb) in enumerate(tqdm(train_loader)):
        points, target, rgb = pc.cuda(), target.cuda(), rgb.cuda()
        print(type(target), target)
        one_hot = F.one_hot(target).half()
        print(one_hot)
        print(rgb)
        print(points.shape, rgb.shape)
        break

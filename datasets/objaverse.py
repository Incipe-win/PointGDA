import os
import random
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader

# template = [
#     "a point cloud model of {}.",
#     "There is a {} in the scene.",
#     "There is the {} in the scene.",
#     "a photo of a {} in the scene.",
#     "a photo of the {} in the scene.",
#     "a photo of one {} in the scene.",
#     "itap of a {}.",
#     "itap of my {}.",
#     "itap of the {}.",
#     "a photo of a {}.",
#     "a photo of my {}.",
#     "a photo of the {}.",
#     "a photo of one {}.",
#     "a photo of many {}.",
#     "a good photo of a {}.",
#     "a good photo of the {}.",
#     "a bad photo of a {}.",
#     "a bad photo of the {}.",
#     "a photo of a nice {}.",
#     "a photo of the nice {}.",
#     "a photo of a cool {}.",
#     "a photo of the cool {}.",
#     "a photo of a weird {}.",
#     "a photo of the weird {}.",
#     "a photo of a small {}.",
#     "a photo of the small {}.",
#     "a photo of a large {}.",
#     "a photo of the large {}.",
#     "a photo of a clean {}.",
#     "a photo of the clean {}.",
#     "a photo of a dirty {}.",
#     "a photo of the dirty {}.",
#     "a bright photo of a {}.",
#     "a bright photo of the {}.",
#     "a dark photo of a {}.",
#     "a dark photo of the {}.",
#     "a photo of a hard to see {}.",
#     "a photo of the hard to see {}.",
#     "a low resolution photo of a {}.",
#     "a low resolution photo of the {}.",
#     "a cropped photo of a {}.",
#     "a cropped photo of the {}.",
#     "a close-up photo of a {}.",
#     "a close-up photo of the {}.",
#     "a jpeg corrupted photo of a {}.",
#     "a jpeg corrupted photo of the {}.",
#     "a blurry photo of a {}.",
#     "a blurry photo of the {}.",
#     "a pixelated photo of a {}.",
#     "a pixelated photo of the {}.",
#     "a black and white photo of the {}.",
#     "a black and white photo of a {}",
#     "a plastic {}.",
#     "the plastic {}.",
#     "a toy {}.",
#     "the toy {}.",
#     "a plushie {}.",
#     "the plushie {}.",
#     "a cartoon {}.",
#     "the cartoon {}.",
#     "an embroidered {}.",
#     "the embroidered {}.",
#     "a painting of the {}.",
#     "a painting of a {}.",
# ]
template = []


class ObjaverseLVIS(Dataset):
    dataset_dir = "objaverse_lvis"

    def __init__(self, root, mode="train", shots=1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        original_split = json.load(open(os.path.join(self.dataset_dir, "lvis.json"), "r"))
        self.mode = mode
        self.shots = shots
        self.num_points = 10000
        self.template = template

        self.category_dict = {}
        for data in original_split:
            cat = data["category"]
            self.category_dict.setdefault(cat, []).append(data)

        self.categories = sorted(self.category_dict.keys())
        self.classnames = self.categories
        self.category2idx = {cat: i for i, cat in enumerate(self.categories)}

        self.split = []
        self._split_dataset()

    def _split_dataset(self):
        for cat in self.categories:
            samples = self.category_dict[cat]

            if self.mode == "train":
                selected = self._sample_with_replacement(samples, self.shots)
                self.split.extend(selected)

            else:
                _, test_samples = self._get_train_test_split(samples, self.shots)
                self.split.extend(test_samples)

    def _sample_with_replacement(self, samples, k):
        if len(samples) == 0:
            return []
        return [random.choice(samples) for _ in range(k)] if k > len(samples) else random.sample(samples, k)

    def _get_train_test_split(self, samples, k):
        train_samples = self._sample_with_replacement(samples, k)

        unique_train_uids = {s["uid"] for s in train_samples}
        test_samples = [s for s in samples if s["uid"] not in unique_train_uids]

        return train_samples, test_samples

    def __getitem__(self, index: int):
        data = np.load(self.split[index]["data_path"], allow_pickle=True).item()
        n = data["xyz"].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data["xyz"][idx]
        rgb = data["rgb"][idx]

        xyz = self.normalize_pc(xyz)

        return (
            torch.from_numpy(xyz).float(),
            self.category2idx[self.split[index]["category"]],
            torch.from_numpy(rgb).float(),
        )

    def __len__(self):
        return len(self.split)

    def normalize_pc(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


if __name__ == "__main__":
    from tqdm import tqdm
    import torch.nn.functional as F

    dataset = ObjaverseLVIS("/workspace/code/deep_learning/PointGDA/data")
    data_loader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=1,
        pin_memory=True,
        shuffle=True,
    )
    print(dataset.classnames)
    print(len(data_loader))
    print(len(dataset.classnames))

    for _, (pc, target, rgb) in enumerate(tqdm(data_loader)):
        points, target, rgb = pc.cuda(), target.cuda(), rgb.cuda()
        print(type(target), target)
        one_hot = F.one_hot(target).half()
        print(one_hot)
        print(rgb)
        print(points.shape, rgb.shape)
        break

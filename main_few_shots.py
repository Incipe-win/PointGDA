import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from torch.autograd import Variable
import numpy as np

from sklearn.covariance import LedoitWolf, OAS, GraphicalLassoCV, GraphicalLasso
from torchvision.transforms import v2
from models import ULIP_models
from collections import OrderedDict


def real_proj(pc, imsize=224):
    pc_views = Realistic_Projection()
    img = pc_views.get_img(pc).cuda()
    img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode="bilinear", align_corners=True)
    return img


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="settings of Tip-Adapter in yaml format")
    parser.add_argument("--output-dir", default="./outputs", type=str, help="output dir")
    parser.add_argument("--test_ckpt_addr", default="", help="the ckpt to test 3d zero shot")
    parser.add_argument("--evaluate_3d", action="store_true", help="eval ulip only")
    parser.add_argument("--npoints", default=2048, type=int, help="number of points used for pre-train and test.")
    args = parser.parse_args()
    return args


def run(cfg, train_loader_cache, clip_weights, clip_model, test_features, test_labels, val_features, val_labels):
    # Parameter Estimation.
    with torch.no_grad():
        # Ours
        vecs = []
        labels = []
        for i in range(cfg["augment_epoch"]):
            for pc, target in tqdm(train_loader_cache):
                pc, target = pc.cuda(), target.cuda()
                pc = clip_model.encode_pc(pc)
                pc = pc / pc.norm(dim=-1, keepdim=True)
                vecs.append(pc)
                labels.append(target)
        vecs = torch.cat(vecs)
        labels = torch.cat(labels)

        # normal distribution
        mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(clip_weights.shape[1])])

        # KS Estimator.
        center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(clip_weights.shape[1])])
        cov_inv = center_vecs.shape[1] * torch.linalg.pinv(
            (center_vecs.shape[0] - 1) * center_vecs.T.cov()
            + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda()
        )

        ps = torch.ones(clip_weights.shape[1]).cuda() * 1.0 / clip_weights.shape[1]
        W = torch.einsum("nd, dc -> cn", mus, cov_inv)
        b = ps.log() - torch.einsum("nd, dc, nc -> n", mus, cov_inv, mus) / 2

        print(
            f"val_features shape is {val_features.shape}, clip_weights shape is {clip_weights.shape}, W shape is {W.shape}, test_features shape is {test_features.shape}"
        )

        # Evaluate
        # Grid search for hyper-parameter alpha
        best_val_acc = 0
        best_alpha = 0.1
        for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            val_logits = 100.0 * val_features.float() @ clip_weights.float() + alpha * (val_features.float() @ W + b)

            acc = cls_acc(val_logits, val_labels)
            if acc > best_val_acc:
                best_val_acc = acc
                best_alpha = alpha

        print("best_val_alpha: %s \t best_val_acc: %s" % (best_alpha, best_val_acc))
        alpha = best_alpha
        test_logits = 100.0 * test_features.float() @ clip_weights.float() + alpha * (test_features.float() @ W + b)
        notune_acc = cls_acc(test_logits, test_labels)
        print("Nonetune acc:", notune_acc)
    return notune_acc


def main():
    # Load config file
    args = get_arguments()
    assert os.path.exists(args.config)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # Load cfg for conditional prompt.
    print("\nRunning configs.")
    print(cfg, "\n")

    ckpt = torch.load(args.test_ckpt_addr, weights_only=False)
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v

    # CLIP
    model = ULIP_models.ULIP_PointBERT(args).cuda()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    notune_accs = {"1": [], "2": [], "3": []}

    seed_list = [1, 2, 3]
    shots_list = [1, 2, 4, 8, 16]

    for seed in seed_list:
        cfg["seed"] = seed
        random.seed(seed)
        torch.manual_seed(seed)

        for shots in shots_list:
            cfg["shots"] = shots

            print("Preparing dataset.")
            dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])

            train_loader_cache = build_data_loader(
                data_source=dataset.train_x, batch_size=64, is_train=True, shuffle=True
            )

            test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, shuffle=False)
            val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, shuffle=False)

            test_features, test_labels = pre_load_features(cfg, "test", model, test_loader)
            val_features, val_labels = pre_load_features(cfg, "val", model, val_loader)

            clip_weights = clip_classifier(cfg, dataset.classnames, dataset.template, model.float())

            notune_acc = run(
                cfg,
                train_loader_cache,
                clip_weights,
                model,
                test_features,
                test_labels,
                val_features,
                val_labels,
            )
            notune_accs[str(cfg["seed"])].append(notune_acc)
    print("Evaluate on dataset:", cfg["dataset"])
    print("Evaluate on seed [1, 2, 3]")
    print("Evaluate on shots [1, 2, 4, 8, 16]")
    print("No tuning:")
    notune = []
    for seed in ["1", "2", "3"]:
        print("seed %s" % seed, notune_accs[str(seed)])
        notune.append(notune_accs[seed])
    notune = torch.tensor(notune)
    print("Average: ", notune.mean(dim=0))


if __name__ == "__main__":
    main()

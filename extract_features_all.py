import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import json
from collections import OrderedDict
from models.uni3d import *
import open_clip
from tokenizer import SimpleTokenizer


def extract_few_shot_feature(cfg, model, train_loader_cache, norm=True):
    cache_keys = []
    cache_values = []
    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg["augment_epoch"]):
            train_features = []
            print("Augment Epoch: {:} / {:}".format(augment_idx, cfg["augment_epoch"]))
            for i, (pc, target, rgb) in enumerate(tqdm(train_loader_cache)):
                pc, rgb = pc.cuda(), rgb.cuda()
                feature = torch.cat((pc, rgb), dim=-1)
                pc_features = get_model(model).encode_pc(feature)  # 100, 3, 224, 224
                train_features.append(pc_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    if norm:
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

    if norm:
        torch.save(cache_keys, cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots.pt")
        torch.save(cache_values, cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots.pt")
    else:
        torch.save(cache_keys, cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots_unnormed.pt")
        torch.save(cache_values, cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots_unnormed.pt")
    return


def extract_few_shot_feature_all(cfg, model, train_loader_cache, norm=True):
    with torch.no_grad():
        # Ours
        vecs = []
        labels = []
        for i in range(cfg["augment_epoch"]):
            for pc, target, rgb in tqdm(train_loader_cache):
                pc, target, rgb = pc.cuda(), target.cuda(), rgb.cuda()
                feature = torch.cat((pc, rgb), dim=-1)
                pc_features = get_model(model).encode_pc(feature)
                if norm:
                    pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
                vecs.append(pc_features)
                labels.append(target)
        vecs = torch.cat(vecs)
        labels = torch.cat(labels)

        if norm:
            torch.save(vecs, cfg["cache_dir"] + "/" + f"{k}_vecs_f.pt")
            torch.save(labels, cfg["cache_dir"] + "/" + f"{k}_labels_f.pt")
        else:
            torch.save(vecs, cfg["cache_dir"] + "/" + f"{k}_vecs_f_unnormed.pt")
            torch.save(labels, cfg["cache_dir"] + "/" + f"{k}_labels_f_unnormed.pt")


def extract_val_test_feature(cfg, split, model, loader, norm=True):
    features, labels = [], []
    with torch.no_grad():
        for i, (pc, target, rgb) in enumerate(tqdm(loader)):
            pc, target, rgb = pc.cuda(), target.cuda(), rgb.cuda()
            feature = torch.cat((pc, rgb), dim=-1)
            pc_features = get_model(model).encode_pc(feature)
            if norm:
                pc_features /= pc_features.norm(dim=-1, keepdim=True)
            features.append(pc_features)
            labels.append(target)
    features, labels = torch.cat(features), torch.cat(labels)
    if norm:
        torch.save(features, cfg["cache_dir"] + "/" + split + "_f.pt")
        torch.save(labels, cfg["cache_dir"] + "/" + split + "_l.pt")
    else:
        torch.save(features, cfg["cache_dir"] + "/" + split + "_f_unnormed.pt")
        torch.save(labels, cfg["cache_dir"] + "/" + split + "_l_unnormed.pt")
    return


def extract_text_feature_all(cfg, classnames, prompt_paths, clip_model, template, norm=True):
    tokenizer = SimpleTokenizer()
    prompts = []
    for prompt_path in prompt_paths:
        f = open(prompt_path)
        prompts.append(json.load(f))
    with torch.no_grad():
        clip_weights = []
        min_len = 1000
        for classname in classnames:
            # Tokenize the prompts
            template_texts = [t.format(classname) for t in template]

            texts = template_texts
            for prompt in prompts:
                texts += prompt[classname]

            texts_token = tokenizer(texts)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            if norm:
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            min_len = min(min_len, len(class_embeddings))
            clip_weights.append(class_embeddings)

        for i in range(len(clip_weights)):
            clip_weights[i] = clip_weights[i][:min_len]

        clip_weights = torch.stack(clip_weights, dim=0).cuda()
        print(clip_weights.shape)

    if norm:
        torch.save(clip_weights, cfg["cache_dir"] + "/text_weights_cupl_t_all.pt")
    else:
        torch.save(clip_weights, cfg["cache_dir"] + "/text_weights_cupl_t_all_unnormed.pt")
    return


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="create_uni3d", type=str)
    parser.add_argument("--npoints", default=2048, type=int, help="number of points used for pre-train and test.")
    parser.add_argument("--group-size", type=int, default=64, help="Pointcloud Transformer group size.")
    parser.add_argument("--num-group", type=int, default=512, help="Pointcloud Transformer number of groups.")
    parser.add_argument("--pc-encoder-dim", type=int, default=512, help="Pointcloud Transformer encoder dimension.")
    parser.add_argument(
        "--clip-model",
        type=str,
        default="RN50",
        help="Name of the vision and text backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pc-model",
        type=str,
        default="RN50",
        help="Name of pointcloud backbone to use.",
    )
    parser.add_argument("--pc-feat-dim", type=int, default=768, help="Pointcloud feature dimension.")
    parser.add_argument("--embed-dim", type=int, default=1024, help="teacher embedding dimension.")
    parser.add_argument("--ckpt_path", default="", help="the ckpt to test 3d zero shot")
    parser.add_argument("--evaluate_3d", action="store_true", help="eval ulip only")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    modelnet40 = "/workspace/code/deep_learning/PointGDA/prompt/modelnet40.json"
    scanobjectnn = "/workspace/code/deep_learning/PointGDA/prompt/scanobjectnn.json"
    args = get_arguments()
    for seed in [1, 2, 3]:
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained)

        model = create_uni3d(args).cuda()
        checkpoint = torch.load(args.ckpt_path)
        sd = checkpoint["module"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        clip_model.eval()

        all_dataset = [
            "modelnet40",
            "scanobjectnn",
        ]
        k_shot = [1, 2, 4, 8, 16]
        norm = True

        data_path = "data"
        for set in all_dataset:
            cfg = yaml.load(open("configs/{}.yaml".format(set), "r"), Loader=yaml.Loader)

            cache_dir = os.path.join(f"./caches/{args.model}/{seed}", cfg["dataset"])
            os.makedirs(cache_dir, exist_ok=True)
            cfg["cache_dir"] = cache_dir
            cfg["backbone"] = args.model
            cfg["seed"] = seed
            cfg["dataset"] = set

            for k in k_shot:

                random.seed(seed)
                torch.manual_seed(seed)

                cfg["shots"] = k
                print(cfg)
                dataset = build_dataset(set, data_path, k)
                val_loader = build_data_loader(data_source=dataset.val, batch_size=32, is_train=False, shuffle=False)
                test_loader = build_data_loader(data_source=dataset.test, batch_size=32, is_train=False, shuffle=False)

                train_loader_cache = build_data_loader(
                    data_source=dataset.train_x,
                    batch_size=32,
                    is_train=True,
                    shuffle=True,
                )

                # Construct the cache model by few-shot training set
                print("\nConstructing cache model by few-shot visual features and labels.")
                extract_few_shot_feature(cfg, model, train_loader_cache)

            # Extract val/test features
            print("\nLoading visual features and labels from val and test set.")
            extract_val_test_feature(cfg, "val", model, val_loader, norm=norm)
            extract_val_test_feature(cfg, "test", model, test_loader, norm=norm)

            # [dataset.cupl_path, dataset.waffle_path, dataset.DCLIP_path]
            if set == "modelnet40":
                extract_text_feature_all(cfg, dataset.classnames, [modelnet40], clip_model, dataset.template, norm=norm)
            else:
                extract_text_feature_all(
                    cfg, dataset.classnames, [scanobjectnn], clip_model, dataset.template, norm=norm
                )

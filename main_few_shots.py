import os
import random
import argparse
import yaml
import torch
from utils import *
from collections import OrderedDict
from PointGDA import PointGDA


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="settings of Tip-Adapter in yaml format")
    parser.add_argument("--shot", dest="shot", type=int, default=1, help="shots number")
    parser.add_argument("--seed", dest="seed", type=int, default=1, help="seed")
    parser.add_argument("--dbg", dest="dbg", type=float, default=0, help="debug mode")
    args = parser.parse_args()
    return args


def main():
    # Load config file
    args = get_arguments()
    assert os.path.exists(args.config)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg["shots"] = args.shot
    cfg["seed"] = args.seed
    cfg["dbg"] = args.dbg
    print("shots", cfg["shots"])
    print("seed", cfg["seed"])
    print("dbg", cfg["dbg"])

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    cache_dir = os.path.join(f'./caches/create_uni3d/{cfg["seed"]}/{cfg["dataset"]}')
    os.makedirs(cache_dir, exist_ok=True)
    cfg["cache_dir"] = cache_dir
    print(cfg)

    # Prepare dataset
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights_cupl_all = torch.load(cfg["cache_dir"] + "/text_weights_cupl_t_all.pt", weights_only=False)
    cate_num, prompt_cupl_num, dim = clip_weights_cupl_all.shape
    print(f"cate_num is {cate_num}, prompt_cupl_num is {prompt_cupl_num}, dim is {dim}")
    clip_weights_cupl = clip_weights_cupl_all.mean(dim=1).t()
    clip_weights_cupl = clip_weights_cupl / clip_weights_cupl.norm(dim=0, keepdim=True)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = load_few_shot_feature(cfg)
    cache_keys = cache_keys + 0.01 * torch.randn_like(cache_keys)  # 添加高斯噪声

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = loda_val_test_feature(cfg, "val")

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    if cfg["dataset"] == "objaverse":
        test_features, test_labels = loda_val_test_feature(cfg, "val")
    else:
        test_features, test_labels = loda_val_test_feature(cfg, "test")

    # ------------------------------------------ Fusion ------------------------------------------
    image_weights_all = torch.stack([cache_keys.t()[torch.argmax(cache_values, dim=1) == i] for i in range(cate_num)])
    image_weights = image_weights_all.mean(dim=1)
    image_weights = image_weights / image_weights.norm(dim=1, keepdim=True)

    clip_weights_IGT, matching_score = image_guide_text(cfg, clip_weights_cupl_all, image_weights, return_matching=True)
    clip_weights_IGT = clip_weights_IGT.t()
    metric = {}
    acc_free = PointGDA(
        cfg,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights_IGT,
        clip_weights_cupl_all,
        matching_score,
        grid_search=False,
        is_print=True,
    )
    metric["TIMO"] = acc_free
    print(f"\nTIMO: {metric['TIMO']:.4f}")  # 确保输出格式可被正则匹配

    # TIMO-S
    # clip_weights_IGT, matching_score = image_guide_text_search(
    #     cfg, clip_weights_cupl_all, val_features, val_labels, image_weights
    # )
    # acc_free = PointGDA(
    #     cfg,
    #     val_features,
    #     val_labels,
    #     test_features,
    #     test_labels,
    #     clip_weights_IGT,
    #     clip_weights_cupl_all,
    #     matching_score,
    #     grid_search=True,
    #     n_quick_search=10,
    #     is_print=True,
    # )
    # metric["TIMO_S"] = acc_free


if __name__ == "__main__":
    main()

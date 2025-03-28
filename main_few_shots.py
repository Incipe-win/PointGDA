import os
import random
import argparse
import yaml
import torch
from utils import *
from collections import OrderedDict
from PointGDA import PointGDA, PointGDA_F
from models.uni3d import *
from datasets.objaverse import ObjaverseLVIS
from torch.utils.data import DataLoader
from datasets import build_dataset
from datasets.utils import build_data_loader


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="settings of Tip-Adapter in yaml format")
    parser.add_argument("--shot", dest="shot", type=int, default=1, help="shots number")
    parser.add_argument("--seed", dest="seed", type=int, default=1, help="seed")
    parser.add_argument("--dbg", dest="dbg", type=float, default=0, help="debug mode")
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
    # acc_free = PointGDA(
    #     cfg,
    #     val_features,
    #     val_labels,
    #     test_features,
    #     test_labels,
    #     clip_weights_IGT,
    #     clip_weights_cupl_all,
    #     matching_score,
    #     grid_search=False,
    #     is_print=True,
    # )
    # metric["TIMO"] = acc_free
    # print(f"\nTIMO: {metric['TIMO']:.4f}")  # 确保输出格式可被正则匹配

    model = create_uni3d(args).cuda()
    checkpoint = torch.load(args.ckpt_path, weights_only=False)
    sd = checkpoint["module"]
    if next(iter(sd.items()))[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    if cfg["dataset"] == "objaverse":
        train_dataset = ObjaverseLVIS(cfg["root_path"], "train", cfg["shots"])
        train_loader_F = DataLoader(train_dataset, num_workers=8, batch_size=4, shuffle=False)
    else:
        dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
        train_loader_F = build_data_loader(
            data_source=dataset.train_x,
            batch_size=32,
            is_train=True,
            shuffle=False,
        )

    acc_free = PointGDA_F(
        cfg,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights_IGT,
        clip_weights_cupl_all,
        matching_score,
        model,
        train_loader_F,
        grid_search=False,
        is_print=True,
    )
    metric["GDA_F"] = acc_free
    print(f"\nGDA_F: {metric['GDA_F']:.4f}")  # 确保输出格式可被正则匹配

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

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
from proj import Realistic_Projection
import torchvision.transforms as transforms
from best_param import best_prompt_weight


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


class Textual_Encoder(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self):
        prompts = best_prompt_weight[
            "{}_{}_test_prompts".format(self.cfg["dataset"].lower(), self.cfg["backbone_name"])
        ]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        text_feat = self.clip_model.encode_text(prompts).repeat(1, self.cfg["num_views"])
        return text_feat


def clip_classifier(cfg, classnames, template, clip_model):
    textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
    text_feat = textual_encoder()
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.t()


def real_proj(pc, imsize=224):
    pc_views = Realistic_Projection()
    img = pc_views.get_img(pc).cuda()
    img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode="bilinear", align_corners=True)
    return img


def save_projection_images(images, batch_idx, base_dir="projected_images"):
    import os
    from PIL import Image

    batch_size, num_views = images.shape[:2]
    batch_dir = os.path.join(base_dir, f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)

    for point_idx in range(batch_size):
        # 创建点云专属目录
        pc_dir = os.path.join(batch_dir, f"point_{point_idx}")
        os.makedirs(pc_dir, exist_ok=True)

        # 生成2x5缩略图拼贴
        collage = Image.new("RGB", (224 * 5, 224 * 2))  # 创建空白画布

        for view_idx in range(num_views):
            # 提取单张图像张量
            img_tensor = images[point_idx, view_idx]

            # 转换张量为PIL图像[4](@ref)
            img_pil = transforms.ToPILImage()(img_tensor)

            # 保存原始视图
            img_pil.save(os.path.join(pc_dir, f"view_{view_idx}.jpg"))

            # 生成缩略图并拼贴
            thumb = img_pil.resize((224, 224))  # 保持原尺寸或调整为112x112
            x = (view_idx % 5) * 224  # 列坐标
            y = (view_idx // 5) * 224  # 行坐标
            collage.paste(thumb, (x, y))

        # 保存拼贴图
        collage.save(os.path.join(pc_dir, "collage.jpg"))


def pre_load_features(cfg, split, clip_model, loader, preprocess, norm=True):
    features, labels = [], []
    view_weights = torch.Tensor(
        best_prompt_weight["{}_{}_test_weights".format(cfg["dataset"].lower(), cfg["backbone_name"])]
    ).cuda()

    with torch.no_grad():
        for i, (pc, target) in enumerate(tqdm(loader)):
            pc, target = pc.cuda(), target.cuda()
            images = real_proj(pc).type(clip_model.dtype)

            # images_visual = images.view(-1, cfg["num_views"], 3, 224, 224)
            # save_projection_images(images_visual, i)

            # ViT/B: channel 512
            image_features = clip_model.encode_image(images)
            if norm:
                image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.reshape(-1, cfg["num_views"], 512) * view_weights.reshape(1, -1, 1)
            image_features = image_features.reshape(-1, cfg["num_views"] * 512).type(clip_model.dtype)
            features.append(image_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)
    return features, labels


def build_cache_model(cfg, clip_model, train_loader_cache):
    cache_keys = []
    cache_values = []

    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg["augment_epoch"]):
            train_features = []

            print("Augment Epoch: {:} / {:}".format(augment_idx, cfg["augment_epoch"]))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.cuda()
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

    return cache_keys, cache_values


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg["search_hp"] == True:

        beta_list = [
            i * (cfg["search_scale"][0] - 0.1) / cfg["search_step"][0] + 0.1 for i in range(cfg["search_step"][0])
        ]
        alpha_list = [
            i * (cfg["search_scale"][1] - 0.1) / cfg["search_step"][1] + 0.1 for i in range(cfg["search_step"][1])
        ]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100.0 * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

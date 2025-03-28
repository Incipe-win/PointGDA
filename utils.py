import torch
import torch.nn.functional as F
import torch.nn as nn
import os


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def load_text_feature(cfg):
    save_path = cfg["cache_dir"] + "/text_weights_gpt_t.pt"
    clip_weights = torch.load(save_path, weights_only=False)
    return clip_weights


def load_few_shot_feature(cfg, norm=True):
    if norm:
        cache_keys = torch.load(cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots.pt", weights_only=False)
        cache_values = torch.load(cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots.pt", weights_only=False)
    else:
        cache_keys = torch.load(
            cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots_unnormed.pt", weights_only=False
        )
        cache_values = torch.load(
            cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots_unnormed.pt", weights_only=False
        )
    return cache_keys, cache_values


def loda_val_test_feature(cfg, split, norm=True):
    if norm:
        features = torch.load(cfg["cache_dir"] + "/" + split + "_f.pt", weights_only=False)
        labels = torch.load(cfg["cache_dir"] + "/" + split + "_l.pt", weights_only=False)
    else:
        features = torch.load(cfg["cache_dir"] + "/" + split + "_f_unnormed.pt", weights_only=False)
        labels = torch.load(cfg["cache_dir"] + "/" + split + "_l_unnormed.pt", weights_only=False)
    return features, labels


# t_features [c,p,d]
# s_features [c,n,d] or [c,d]
def image_guide_text(cfg, t_features, s_features, gamma=-1, return_weights=False, return_matching=False):
    t_features = t_features / t_features.norm(dim=-1, keepdim=True)

    if gamma == -1:
        if cfg["dataset"] == "imagenet":
            gamma = 1
        elif cfg["dataset"] == "oxford_flowers":
            gamma = 100
        else:
            gamma = 50

    cate_num, prompt_num, feat_dim = t_features.shape  # c, p, d
    if len(s_features.shape) == 3:
        s_features = s_features.mean(dim=1)  # c,d
        s_features = s_features / s_features.norm(dim=-1, keepdim=True)
    weights = torch.ones(cate_num, prompt_num).to(t_features.dtype).to(t_features.device)  # c, p
    s_features = s_features.to(t_features.dtype)
    t_features = t_features / t_features.norm(dim=-1, keepdim=True)

    matching_score = []
    for c in range(cate_num):
        # weights[c:c+1] # 1, p
        # t_features[c] # p, d
        # s_features[c:c+1] # 1, d
        weights[c] = (s_features[c : c + 1] @ t_features[c].t()).squeeze(dim=0)
        matching_score.append(weights[c].clone())
        weights[c] = weights[c] / weights[c].norm()
        weights[c] = F.softmax(weights[c] * gamma, dim=0)
    matching_score = torch.stack(matching_score, dim=0)  # N, P

    for weights in [weights]:
        normed_weights = weights
        normed_clip_weights = torch.einsum("cp, cpd-> cd", normed_weights, t_features)
        normed_clip_weights = normed_clip_weights / normed_clip_weights.norm(dim=-1, keepdim=True)

    if return_matching:
        return normed_clip_weights, matching_score
    elif return_weights:
        return normed_clip_weights, normed_weights
    else:
        return normed_clip_weights


def vec_sort(vecs_t, matching_score):
    cate_num, prompt_num, dim = vecs_t.shape  # N,P,D

    weights, sorted_idx = torch.topk(matching_score, k=prompt_num, dim=-1)
    sort_vecs_t = []
    for c in range(cate_num):
        sort_vecs_t.append(vecs_t[c][sorted_idx[c]].clone())
    sort_vecs_t = torch.stack(sort_vecs_t, dim=0)

    if len(sort_vecs_t.shape) == 2:
        sort_vecs_t = sort_vecs_t.unsqueeze(1)

    return sort_vecs_t, weights


def image_guide_text_search(cfg, clip_weights_cupl_all, val_features, val_labels, image_weights):
    best_acc = 0
    best_gamma = 0
    for gamma in range(5, 101, 5):
        clip_weights_cupl_IGT, matching_score = image_guide_text(
            cfg, clip_weights_cupl_all, image_weights, return_matching=True, gamma=gamma
        )
        clip_weights_cupl_IGT = clip_weights_cupl_IGT.t()  # D, C

        val_logits = val_features @ clip_weights_cupl_IGT  # N, C
        acc = (val_logits.argmax(-1) == val_labels).sum() / len(val_labels)

        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma
    print("best_gamma:", best_gamma)
    clip_weights_cupl_IGT, matching_score = image_guide_text(
        cfg, clip_weights_cupl_all, image_weights, return_matching=True, gamma=best_gamma
    )
    clip_weights_cupl_IGT = clip_weights_cupl_IGT.t()
    return clip_weights_cupl_IGT, matching_score


# def cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=False, training_free=False):

#     feat_dim, cate_num = clip_weights.shape
#     text_feat = clip_weights.t().unsqueeze(1)
#     cache_feat = cache_keys.reshape(cate_num, cfg["shots"] * cfg["augment_epoch"], feat_dim)

#     save_path = f"caches/create_uni3d/{cfg['seed']}/{cfg['dataset']}"
#     save_file = "{}/criterion_{}_{}shot.pt".format(save_path, cfg["backbone"].replace("/", ""), cfg["shots"])

#     if os.path.exists(save_file):
#         print("Loading criterion...")
#         sim = torch.load(save_file, weights_only=False)
#     elif only_use_txt:
#         print("Calculating criterion...")

#         feats = text_feat.squeeze()

#         sim_sum = torch.zeros((feat_dim)).cuda()
#         count = 0
#         for i in range(cate_num):
#             for j in range(cate_num):
#                 if i != j:
#                     sim_sum += feats[i, :] * feats[j, :]
#                     count += 1
#         sim = sim_sum / count
#         torch.save(sim, save_file)
#     else:
#         print("Calculating criterion...")

#         feats = torch.cat([text_feat, cache_feat], dim=1)
#         samp_num = feats.shape[1]

#         sim_sum = torch.zeros((feat_dim)).cuda()
#         count = 0
#         for i in range(cate_num):
#             for j in range(cate_num):
#                 for m in range(samp_num):
#                     for n in range(samp_num):
#                         if i != j:
#                             sim_sum += feats[i, m, :] * feats[j, n, :]
#                             count += 1
#         sim = sim_sum / count
#         torch.save(sim, save_file)

#     criterion = (-1) * cfg["w"][0] * sim + cfg["w"][1] * torch.var(clip_weights, dim=1)

#     if training_free:
#         _, indices = torch.topk(criterion, k=cfg["training_free_feat_num"])
#     else:
#         _, indices = torch.topk(criterion, k=cfg["training_feat_num"])
#     return indices


def cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=False, training_free=False):
    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1)
    cache_feat = cache_keys.reshape(cate_num, cfg["shots"] * cfg["augment_epoch"], feat_dim)

    save_path = f"caches/create_uni3d/{cfg['seed']}/{cfg['dataset']}"
    save_file = f"{save_path}/criterion_{cfg['backbone'].replace('/', '')}_{cfg['shots']}shot.pt"

    if os.path.exists(save_file):
        print("Loading criterion...")
        sim = torch.load(save_file, weights_only=False)
    elif only_use_txt:
        print("Calculating criterion (text only)...")
        feats = text_feat.squeeze()  # Shape: (cate_num, feat_dim)

        # Vectorized computation replacing double loops
        sum_feat = feats.sum(dim=0)
        sum_sq_feat = (feats**2).sum(dim=0)
        sim_sum = sum_feat**2 - sum_sq_feat
        count = cate_num * (cate_num - 1)
        sim = sim_sum / count

        torch.save(sim, save_file)
    else:
        print("Calculating criterion with cache...")
        feats = torch.cat([text_feat, cache_feat], dim=1)  # Shape: (cate_num, samp_num, feat_dim)
        samp_num = feats.shape[1]

        # Vectorized computation replacing quadruple loops
        sum_feats = feats.sum(dim=1)  # Sum over samples, shape: (cate_num, feat_dim)
        total_sum = (sum_feats.sum(dim=0)) ** 2
        diag_sum = (sum_feats**2).sum(dim=0)
        sim_sum = total_sum - diag_sum
        count = cate_num * (cate_num - 1) * (samp_num**2)
        sim = sim_sum / count

        torch.save(sim, save_file)

    # Original post-processing remains unchanged
    criterion = (-1) * cfg["w"][0] * sim + cfg["w"][1] * torch.var(clip_weights, dim=1)
    k = cfg["training_free_feat_num"] if training_free else cfg["training_feat_num"]
    _, indices = torch.topk(criterion, k=k)
    return indices


class GDA_Training(nn.Module):
    def __init__(self, cfg, clip_weights, model, cache_keys):
        super(GDA_Training, self).__init__()
        self.shots = cfg["shots"] * cfg["augment_epoch"]
        self.feat_dim, self.cate_num = clip_weights.shape

        self.value_weights = nn.Parameter(
            torch.ones([self.cate_num * cfg["shots"] * cfg["augment_epoch"], 1]).half().cuda(), requires_grad=True
        )
        self.indices = cal_criterion(cfg, clip_weights, cache_keys)

        self.res = nn.Parameter(
            torch.zeros([self.cate_num, cfg["training_feat_num"]]).half().cuda(), requires_grad=True
        )
        self.feat_num = cfg["training_feat_num"]

    def forward(self, cache_keys, clip_weights, cache_values):
        res_keys = self.res.unsqueeze(1).repeat(1, self.shots, 1).reshape(-1, self.feat_num)
        new_cache_keys = cache_keys.clone()
        new_cache_keys = new_cache_keys.reshape(-1, self.feat_dim)
        new_cache_keys[:, self.indices] = new_cache_keys[:, self.indices] + res_keys

        res_text = self.res.t()
        new_clip_weights = clip_weights.clone()
        new_clip_weights[self.indices, :] = clip_weights[self.indices, :] + res_text
        new_cache_values = cache_values * self.value_weights

        return new_cache_keys, new_clip_weights, new_cache_values


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * (1.0 - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

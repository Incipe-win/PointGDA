import torch
import torch.nn.functional as F


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
    clip_weights = torch.load(save_path)
    return clip_weights


def load_few_shot_feature(cfg, norm=True):
    if norm:
        cache_keys = torch.load(cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots.pt")
        cache_values = torch.load(cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots.pt")
    else:
        cache_keys = torch.load(cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots_unnormed.pt")
        cache_values = torch.load(cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots_unnormed.pt")
    return cache_keys, cache_values


def loda_val_test_feature(cfg, split, norm=True):
    if norm:
        features = torch.load(cfg["cache_dir"] + "/" + split + "_f.pt")
        labels = torch.load(cfg["cache_dir"] + "/" + split + "_l.pt")
    else:
        features = torch.load(cfg["cache_dir"] + "/" + split + "_f_unnormed.pt")
        labels = torch.load(cfg["cache_dir"] + "/" + split + "_l_unnormed.pt")
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

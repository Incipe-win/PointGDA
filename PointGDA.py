from utils import *


def GDA(vecs, labels, clip_weights, val_features, val_labels, alpha_shift=False):
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

    # Evaluate
    # Grid search for hyper-parameter alpha
    best_val_acc = 0
    best_alpha = 0.1
    for alpha in [10**i for i in range(-4, 5)]:  # [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        if alpha_shift:
            val_logits = alpha * val_features.float() @ clip_weights.float() + val_features.float() @ W + b
        else:
            val_logits = 100.0 * val_features.float() @ clip_weights.float() + alpha * (val_features.float() @ W + b)

        acc = cls_acc(val_logits, val_labels)
        if acc > best_val_acc:
            best_val_acc = acc
            best_alpha = alpha

    print("best_val_alpha: %s \t best_val_acc: %s" % (best_alpha, best_val_acc))
    alpha = best_alpha
    return alpha, W, b, best_val_acc


def PointGDA(
    cfg,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
    clip_weights_all,
    matching_score,
    vecs_labels=None,
    grid_search=False,
    n_quick_search=-1,
    is_print=False,
):

    best_val_acc = 0
    best_alpha = 0.1

    with torch.no_grad():
        # Image Vecs
        if vecs_labels is None:
            vecs_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_vecs_f.pt").float()
            labels_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_labels_f.pt").float()
        else:
            vecs_v, labels_v = vecs_labels[0], vecs_labels[1]

        vecs_t = clip_weights_all.clone().float()  # c, n, d
        vecs_t, weights = vec_sort(vecs_t, matching_score)
        (
            cate_num,
            prompt_num,
            _,
        ) = vecs_t.shape
        vecs_c, labels_c = vecs_v, labels_v

        if grid_search:
            if n_quick_search != -1:
                beta_list = [int(t) for t in torch.linspace(1, prompt_num * 2, n_quick_search)]
            else:
                beta_list = range(1, prompt_num * 2)
        else:
            beta_list = [prompt_num]

        for beta in beta_list:
            beta = beta + 1 if beta == 0 else beta

            sliced_vecs_t = vecs_t.repeat(1, 2, 1)[:, :beta, :]  # c, s, d
            sliced_weights = weights.repeat(1, 2)[:, :beta]  # c, s

            # weight for instance based transfer
            sliced_vecs_t = sliced_vecs_t * sliced_weights.unsqueeze(-1)

            sliced_vecs_t = sliced_vecs_t.reshape(cate_num * beta, -1)
            tmp = torch.tensor(range(cate_num)).unsqueeze(1).repeat(1, beta)
            sliced_labels_t = tmp.flatten().to(sliced_vecs_t.device)

            # Instance based transfer
            vecs_c = torch.cat([sliced_vecs_t, vecs_v])
            labels_c = torch.cat([sliced_labels_t, labels_v])

            alpha, W, b, val_acc = GDA(vecs_c, labels_c, clip_weights, val_features, val_labels, alpha_shift=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_beta = beta
                best_alpha = alpha
                best_weights = [W.clone(), b.clone()]

        alpha = best_alpha
        test_logits = alpha * test_features.float() @ clip_weights.float() + (
            test_features.float() @ best_weights[0] + best_weights[1]
        )
        acc = cls_acc(test_logits, test_labels)

        if is_print:
            print("best_val_alpha: %s \t best_val_acc: %s" % (best_alpha, best_val_acc))
            print("best_beta:", best_beta)
            print("training-free acc:", acc)
            print()

    return acc

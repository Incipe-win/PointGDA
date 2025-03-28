from utils import *
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from sklearn.decomposition import PCA
from scipy.stats import normaltest
from sklearn.cluster import KMeans
from tqdm import tqdm


class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar  # 正则化系数

    def fit(self, X):
        n_samples, n_features = X.shape
        device = X.device

        # 初始化参数（添加随机扰动）
        self.weights_ = torch.ones(self.n_components, device=device) / self.n_components
        X_np = X.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_components, init="k-means++", random_state=42).fit(X_np)
        self.means_ = torch.tensor(kmeans.cluster_centers_, device=device).float()
        self.covariances_ = torch.stack(
            [
                torch.eye(n_features, device=device) + self.reg_covar * torch.randn(n_features, device=device)
                for _ in range(self.n_components)
            ]
        )

        prev_lower_bound = -np.inf

        for _ in range(self.max_iter):
            # E-step: 计算对数概率（数值稳定版本）
            log_prob = []
            for k in range(self.n_components):
                try:
                    # 添加正则化确保协方差正定
                    cov = self.covariances_[k] + self.reg_covar * torch.eye(n_features, device=device)
                    mvn = MultivariateNormal(self.means_[k], cov)
                    log_prob_k = mvn.log_prob(X)
                except ValueError:
                    # 如果仍然失败，使用对角线近似
                    log_prob_k = -0.5 * torch.sum(
                        (X - self.means_[k]) ** 2 / (torch.diag(self.covariances_[k]) + self.reg_covar), dim=1
                    )
                log_prob.append(log_prob_k + torch.log(self.weights_[k]))

            weighted_logprob = torch.stack(log_prob, dim=1)
            log_resp = weighted_logprob - torch.logsumexp(weighted_logprob, dim=1, keepdim=True)
            resp = torch.exp(log_resp)

            # M-step: 更新参数
            Nk = resp.sum(dim=0) + 1e-10  # 防止除零
            self.weights_ = Nk / n_samples
            self.means_ = (resp.T @ X) / Nk[:, None]

            # 更新协方差矩阵（添加正则化）
            diff = X[:, None, :] - self.means_
            covs = torch.einsum("nk,nki,nkj->kij", resp, diff, diff) / Nk[:, None, None]
            self.covariances_ = covs + self.reg_covar * torch.eye(n_features, device=device)[None, :, :]

            # 检查收敛
            current_lower_bound = (resp * weighted_logprob).sum()
            if current_lower_bound - prev_lower_bound < self.tol:
                break
            prev_lower_bound = current_lower_bound

        return self


def Optimal_GDA(
    vecs, labels, clip_weights, val_features, val_labels, alpha_shift=False, n_components=2, use_learned_features=True
):
    # 计算每个类的GMM混合均值
    mus = []
    for i in range(clip_weights.shape[1]):
        class_mask = labels == i
        class_data = vecs[class_mask]

        if len(class_data) < n_components:
            mu = torch.mean(class_data, dim=0)
        else:
            # 使用PyTorch自定义GMM
            gmm = GMM(n_components=n_components, reg_covar=1e-3).fit(class_data)
            mu = (gmm.weights_[:, None] * gmm.means_).sum(dim=0)
        mus.append(mu)
    mus = torch.stack(mus)

    # 计算中心向量
    center_vecs = torch.cat([vecs[labels == i] - mus[i] for i in range(clip_weights.shape[1])])

    # 正则化协方差矩阵并求逆
    d = center_vecs.shape[1]
    cov = (center_vecs.T @ center_vecs) / (center_vecs.shape[0] - 1)
    cov_reg = cov + torch.trace(cov) / d * torch.eye(d).cuda()  # 正则化
    cov_inv = d * torch.linalg.pinv(cov_reg)

    # 计算先验概率和线性参数
    ps = torch.ones(clip_weights.shape[1]).cuda() / clip_weights.shape[1]
    W = torch.einsum("nd,dc->cn", mus, cov_inv)
    b = ps.log() - 0.5 * torch.einsum("nd,dc,nc->n", mus, cov_inv, mus)

    # 超参数搜索
    best_val_acc, best_alpha = 0, 0.1
    for alpha in [10**i for i in range(-4, 5)]:
        if alpha_shift:
            val_logits = alpha * val_features @ clip_weights + val_features @ W + b
        else:
            val_logits = 100.0 * val_features @ clip_weights + alpha * (val_features @ W + b)
        acc = cls_acc(val_logits, val_labels)
        if acc > best_val_acc:
            best_val_acc, best_alpha = acc, alpha

    print(f"best_val_alpha: {best_alpha}\tbest_val_acc: {best_val_acc}")
    return best_alpha, W, b, best_val_acc


def GDA(vecs, labels, clip_weights, val_features, val_labels, alpha_shift=False, n_components=2):
    # 计算每个类的GMM混合均值
    mus = []
    for i in range(clip_weights.shape[1]):
        class_mask = labels == i
        class_data = vecs[class_mask].cpu().numpy()  # 转换为numpy
        if len(class_data) < n_components:
            # 样本不足时回退到单高斯
            mu = torch.mean(vecs[class_mask], dim=0)
        else:
            gmm = GaussianMixture(n_components=n_components).fit(class_data)
            # 计算加权平均均值
            weights = gmm.weights_
            means = gmm.means_
            weighted_mu = np.sum(weights[:, np.newaxis] * means, axis=0)
            mu = torch.tensor(weighted_mu, dtype=torch.float32).cuda()
        mus.append(mu)
    mus = torch.stack(mus)

    # 计算中心向量
    center_vecs = torch.cat([vecs[labels == i] - mus[i] for i in range(clip_weights.shape[1])])

    # 正则化协方差矩阵并求逆
    d = center_vecs.shape[1]
    cov = (center_vecs.T @ center_vecs) / (center_vecs.shape[0] - 1)
    cov_reg = cov + torch.trace(cov) / d * torch.eye(d).cuda()  # 正则化
    cov_inv = d * torch.linalg.pinv(cov_reg)

    # 计算先验概率和线性参数
    ps = torch.ones(clip_weights.shape[1]).cuda() / clip_weights.shape[1]
    W = torch.einsum("nd,dc->cn", mus, cov_inv)
    b = ps.log() - 0.5 * torch.einsum("nd,dc,nc->n", mus, cov_inv, mus)

    # 超参数搜索
    best_val_acc, best_alpha = 0, 0.1
    for alpha in [10**i for i in range(-4, 5)]:
        if alpha_shift:
            val_logits = alpha * val_features @ clip_weights + val_features @ W + b
        else:
            val_logits = 100.0 * val_features @ clip_weights + alpha * (val_features @ W + b)
        acc = cls_acc(val_logits, val_labels)
        if acc > best_val_acc:
            best_val_acc, best_alpha = acc, alpha

    print(f"best_val_alpha: {best_alpha}\tbest_val_acc: {best_val_acc}")
    return best_alpha, W, b, best_val_acc


def Native_GDA(vecs, labels, clip_weights, val_features, val_labels, alpha_shift=False):
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


def helper(
    cfg,
    cache_keys,
    cache_values,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
    model,
    train_loader_F,
):
    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num)
    cache_keys = (
        cache_keys.t().reshape(cate_num, cfg["shots"] * cfg["augment_epoch"], feat_dim).reshape(cate_num, -1, feat_dim)
    )

    cfg["w"] = cfg["w_training"]
    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)
    adapter = GDA_Training(cfg, clip_weights, model, cache_keys).cuda()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["lr"], eps=cfg["eps"], weight_decay=1e-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["train_epoch"] * len(train_loader_F))
    Loss = SmoothCrossEntropy(alpha=0.1)

    beta, alpha = cfg["init_beta"], cfg["init_alpha"]
    best_acc, best_epoch = 0.0, 0
    # feat_num = cfg["feat_num"]

    for train_idx in range(cfg["train_epoch"]):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print("Train Epoch: {:} / {:}".format(train_idx, cfg["train_epoch"]))

        for i, (pc, target, rgb) in enumerate(tqdm(train_loader_F)):
            pc, target, rgb = pc.cuda(), target.cuda(), rgb.cuda()
            feature = torch.cat((pc, rgb), dim=-1)
            with torch.no_grad():
                pc_features = get_model(model).encode_pc(feature)
                pc_features /= pc_features.norm(dim=-1, keepdim=True)

            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
            R_fF = pc_features @ new_cache_keys.t()
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100.0 * pc_features @ new_clip_weights
            ape_logits = R_fW + cache_logits * alpha

            loss = Loss(ape_logits, target)

            # 特征一致性损失
            keys_mse = F.mse_loss(new_cache_keys, cache_keys)
            clip_mse = F.mse_loss(new_clip_weights, clip_weights)

            # 参数正则化（可选）
            res_l2 = torch.norm(adapter.res)
            value_weights_l2 = torch.norm(adapter.value_weights)

            # 总损失 = 主损失 + 特征一致性 + 正则化
            loss = (
                loss
                + cfg["keys_mse_weight"] * keys_mse
                + cfg["clip_mse_weight"] * clip_mse
                + cfg["res_l2_weight"] * res_l2
                + cfg["value_weights_l2_weight"] * value_weights_l2
            )

            acc = cls_acc(ape_logits, target)
            correct_samples += acc / 100 * len(ape_logits)
            all_samples += len(ape_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            "LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}".format(
                current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list) / len(loss_list)
            )
        )

        # Eval
        adapter.eval()
        with torch.no_grad():
            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)

            R_fF = val_features @ new_cache_keys.t()
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100.0 * val_features @ new_clip_weights
            ape_logits = R_fW + cache_logits * alpha
        acc = cls_acc(ape_logits, val_labels)

        print("**** APE-T's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter, cfg["cache_dir"] + "/APE-T_" + str(cfg["shots"]) + "shots.pt")

    adapter = torch.load(cfg["cache_dir"] + "/APE-T_" + str(cfg["shots"]) + "shots.pt", weights_only=False)
    return adapter(cache_keys, clip_weights, cache_values)


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
    pca_dim=256,
    q=700,
):
    # if cfg["dataset"] == "objaverse":
    #     pca_dim = 1024
    device = val_features.device
    best_val_acc = 0
    best_alpha = 0.1

    with torch.no_grad():
        # Image Vecs
        if vecs_labels is None:
            vecs_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
            labels_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
        else:
            vecs_v, labels_v = vecs_labels[0], vecs_labels[1]

        print(vecs_v.shape, clip_weights.shape, labels_v.shape)

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

            # # ==================== 新增特征处理逻辑 ====================
            # # 将训练数据转换为numpy处理
            # train_data_np = vecs_c.cpu().numpy()

            # # 1. 标准化每个特征维度
            # means = np.mean(train_data_np, axis=0)
            # stds = np.std(train_data_np, axis=0)
            # stds[stds == 0] = 1.0  # 防止除零

            # # 2. 正态性检验并选择特征
            # p_values = []
            # for i in range(train_data_np.shape[1]):
            #     if len(train_data_np) >= 20:  # 确保足够样本
            #         _, p = normaltest((train_data_np[:, i] - means[i]) / stds[i])
            #         p_values.append(p)
            #     else:
            #         p_values.append(0.0)  # 样本不足时默认p值

            # # 3. 按p值选择top Q维度
            # sorted_indices = np.argsort(p_values)[::-1].copy()  # 降序排列
            # selected_indices = sorted_indices[:q]

            # # 转换为torch张量并移至对应设备
            # means_tensor = torch.tensor(means, dtype=vecs_c.dtype, device=device)
            # stds_tensor = torch.tensor(stds, dtype=vecs_c.dtype, device=device)
            # selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)

            # # 对特征进行标准化和维度选择
            # def process_features(features):
            #     normalized = (features - means_tensor) / stds_tensor
            #     return normalized.index_select(1, selected_indices)

            # vecs_c = process_features(vecs_c)
            # val_features = process_features(val_features)
            # test_features = process_features(test_features)

            # # 调整clip_weights维度
            # clip_weights = clip_weights.index_select(0, selected_indices)
            # # ========================================================

            # # 添加PCA降维逻辑
            # if pca_dim is not None:
            #     # 合并训练数据（vecs_c）和CLIP模板数据（sliced_vecs_t）来拟合PCA
            #     all_train_data = torch.cat([sliced_vecs_t, vecs_v]).cpu().numpy()
            #     pca = PCA(n_components=pca_dim, random_state=42)
            #     pca.fit(all_train_data)

            #     # 对训练数据降维
            #     vecs_c = torch.tensor(pca.transform(vecs_c.cpu().numpy()), device=device)

            #     # 对验证数据降维
            #     val_features = torch.tensor(pca.transform(val_features.cpu().numpy()), device=device)

            #     # 对测试数据降维
            #     test_features = torch.tensor(pca.transform(test_features.cpu().numpy()), device=device)

            #     # 对CLIP权重降维（处理维度转置）
            #     clip_weights_np = clip_weights.cpu().numpy().T  # 转置为 [C, D]
            #     clip_weights_reduced = pca.transform(clip_weights_np)
            #     clip_weights = torch.tensor(clip_weights_reduced.T, device=device)  # 转置回 [pca_dim, C]

            # print(vecs_c.shape, val_features.shape, clip_weights.shape)
            alpha, W, b, val_acc = Optimal_GDA(
                vecs_c,
                labels_c,
                clip_weights,
                val_features,
                val_labels,
                alpha_shift=True,
            )
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


def PointGDA_F(
    cfg,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
    clip_weights_all,
    matching_score,
    model,
    train_loader_F,
    vecs_labels=None,
    grid_search=False,
    n_quick_search=-1,
    is_print=False,
    pca_dim=256,
):
    device = val_features.device
    best_val_acc = 0
    best_alpha = 0.1

    if vecs_labels is None:
        vecs_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
        labels_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
    else:
        vecs_v, labels_v = vecs_labels[0], vecs_labels[1]

    vecs_v = vecs_v.T
    labels_v = F.one_hot(labels_v.long(), num_classes=clip_weights.shape[1]).float()
    vecs_v, clip_weights, labels_v = helper(
        cfg,
        vecs_v,
        labels_v,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights,
        model,
        train_loader_F,
    )
    labels_v = labels_v.argmax(dim=1)
    print(vecs_v.shape, clip_weights.shape, labels_v.shape)

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

        # 添加PCA降维逻辑
        if pca_dim is not None:
            # 合并训练数据（vecs_c）和CLIP模板数据（sliced_vecs_t）来拟合PCA
            all_train_data = torch.cat([sliced_vecs_t, vecs_v]).detach().cpu().numpy()
            pca = PCA(n_components=pca_dim, random_state=42)
            pca.fit(all_train_data)

            # 对训练数据降维
            vecs_c = torch.tensor(pca.transform(vecs_c.detach().cpu().numpy()), device=device)

            # 对验证数据降维
            val_features = torch.tensor(pca.transform(val_features.detach().cpu().numpy()), device=device)

            # 对测试数据降维
            test_features = torch.tensor(pca.transform(test_features.detach().cpu().numpy()), device=device)

            # 对CLIP权重降维（处理维度转置）
            clip_weights_np = clip_weights.detach().cpu().numpy().T  # 转置为 [C, D]
            clip_weights_reduced = pca.transform(clip_weights_np)
            clip_weights = torch.tensor(clip_weights_reduced.T, device=device)  # 转置回 [pca_dim, C]

        # print(vecs_c.shape, val_features.shape, clip_weights.shape)
        alpha, W, b, val_acc = Optimal_GDA(
            vecs_c,
            labels_c,
            clip_weights,
            val_features,
            val_labels,
            alpha_shift=True,
        )
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

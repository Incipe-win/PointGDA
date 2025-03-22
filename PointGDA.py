from utils import *
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
from torch.distributions import MultivariateNormal


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
        self.means_ = X[torch.randperm(n_samples)[: self.n_components]] + 1e-6 * torch.randn(
            self.n_components, n_features, device=device
        )
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


def Optimal_GDA(vecs, labels, clip_weights, val_features, val_labels, alpha_shift=False, n_components=2):
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
            vecs_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
            labels_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
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

            alpha, W, b, val_acc = Optimal_GDA(
                vecs_c, labels_c, clip_weights, val_features, val_labels, alpha_shift=True
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

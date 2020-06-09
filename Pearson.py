import torch
from itertools import combinations
from scipy.special import comb


def pearsonr2d(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        x = np.random.randn(100)
        y = np.random.randn(100)
        sp_corr = scipy.stats.pearsonr(x, y)[0]
        th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x, 1, keepdim=True)
    mean_y = torch.mean(y, 1, keepdim=True)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(xm * ym, dim=1)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1) + 0.0000001
    r_val = r_num / r_den
    return r_val

def pearsonr1d(x, y):
    vx = x.sub(torch.mean(x))
    vy = y.sub(torch.mean(y))
    return torch.sum(vx * vy) / (torch.norm(vx, 2) * torch.norm(vy, 2) + 0.0000001)

def cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def ncl_loss(eta_hat):
    n_models, _, num_classes=eta_hat.shape
    eta_hat_softmax = torch.softmax(eta_hat,2)

    ncl_val = 0
    for i in range(n_models):
        for j in range(n_models):
            if j == i:
                continue

            else:
                pairwise_ncl_loss = cross_entropy(eta_hat[j],eta_hat_softmax[i])

            ncl_val += pairwise_ncl_loss
    ncl_val = -ncl_val /(n_models*(n_models-1))
    return ncl_val


def pearson_corr_loss(eta_hat, labels, threshold=0.9, has_sofmax=True):
    n_models, _, num_classes = eta_hat.shape
    if n_models < 2:
        return torch.tensor(0)

    orig_mask = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    mask = (1 - orig_mask).type(torch.bool)

    if has_sofmax:
        wrong_classes_outputs = [torch.masked_select(eta_hat[i], mask).reshape((-1, num_classes - 1))
                                 for i in
                                 range(len(eta_hat))]

        wrong_classes_indicator = [
            torch.masked_select(eta_hat[i], orig_mask.type(torch.bool)).reshape(
                (-1, 1)) - torch.masked_select(eta_hat[i], mask).reshape(
                (-1, num_classes - 1)) - threshold
            for i in range(len(eta_hat))]
    else:
        wrong_classes_outputs = [torch.masked_select(torch.softmax(eta_hat[i], 1), mask).reshape((-1, num_classes - 1))
                                 for i in
                                 range(len(eta_hat))]

        wrong_classes_indicator = [
            torch.masked_select(torch.softmax(eta_hat[i], 1), orig_mask.type(torch.bool)).reshape(
                (-1, 1)) - torch.masked_select(torch.softmax(eta_hat[i], 1), mask).reshape((-1, num_classes - 1)) - threshold
            for i in range(len(eta_hat))]

    wrong_classes_indicator = [torch.relu(-torch.min(wrong_classes_indicator[i], 1).values) for i in
                               range(len(eta_hat))]

    # ganovich - change to combination
    pearson_corr = 0
    for i, j in combinations(range(n_models), 2):
        relevant_locs = wrong_classes_indicator[i] + wrong_classes_indicator[j]
        pairwise_corr = pearsonr2d(wrong_classes_outputs[i], wrong_classes_outputs[j])
        pairwise_corr = pairwise_corr[relevant_locs > 0.]
        relevant_locs = relevant_locs[relevant_locs > 0.]

        pairwise_corr = pairwise_corr.sum() / (relevant_locs.shape[0] + 0.0001)
        pearson_corr += pairwise_corr

    pearson_corr /= comb(n_models, 2)
    return pearson_corr


def pearson_corr_loss_multicalss(eta_hat, labels, threshold=0.9):
    n_models, n_batch, num_classes = eta_hat.shape
    if n_models < 2:
        return torch.tensor(0)

    orig_mask = labels.type(torch.bool)
    mask = ~orig_mask
    sample_n_wrong = mask.sum(axis=-1).int().cpu().numpy()
    sample_n_right = tuple(num_classes - sample_n_wrong)
    sample_n_wrong = tuple(sample_n_wrong)

    wrong_classes_outputs = [torch.masked_select(eta_hat[i], mask).split_with_sizes(sample_n_wrong)
                             for i in
                             range(len(eta_hat))]
    right_classes_outputs = [torch.masked_select(eta_hat[i], orig_mask).split_with_sizes(sample_n_right)
                             for i in
                             range(len(eta_hat))]

    # in multi class some samples can have... all classes!
    right_classes_outputs = [[y for j,y in enumerate(x) if len(wrong_classes_outputs[i][j]) > 0] for i,x in
                             enumerate(right_classes_outputs)]
    wrong_classes_outputs = [[y for j, y in enumerate(x) if len(wrong_classes_outputs[i][j]) > 0] for i, x in
                             enumerate(wrong_classes_outputs)]

    wrong_classes_indicator = []
    for model_res_w, model_res_r in zip(wrong_classes_outputs, right_classes_outputs):
        model_indicator = []
        for sample_w, sample_r in zip(model_res_w, model_res_r):
            # in multi class some samples can have... everything!
            indicator = sample_w - sample_r.mean() - threshold
            indicator = torch.relu(-torch.min(indicator))
            model_indicator.append(indicator)
        wrong_classes_indicator.append(torch.stack(model_indicator))

    pearson_corr = 0
    for i, j in combinations(range(n_models), 2):
        relevant_locs = wrong_classes_indicator[i] + wrong_classes_indicator[j]
        pairwise_corr = []
        for sample_i, sapmle_j in zip(wrong_classes_outputs[i], wrong_classes_outputs[j]):
            pairwise_corr.append(pearsonr1d(sample_i, sapmle_j))
        pairwise_corr = torch.stack(pairwise_corr)
        pairwise_corr = pairwise_corr[relevant_locs > 0.]
        relevant_locs = relevant_locs[relevant_locs > 0.]

        pairwise_corr = pairwise_corr.sum() / (relevant_locs.shape[0] + 0.0001)
        pearson_corr += pairwise_corr

    pearson_corr /= comb(n_models, 2)
    return pearson_corr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()
        # m.requires_grad = False
        m.track_running_stats = False

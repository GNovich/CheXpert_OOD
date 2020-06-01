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


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()
        # m.requires_grad = False
        m.track_running_stats = False

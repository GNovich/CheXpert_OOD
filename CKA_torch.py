"""
    Gal Novich

    torch ver inspired by:
        "Similarity of Neural Network Representations Revisited"
        https://arxiv.org/pdf/1905.00414.pdf

        "Ensemble Soft-Margin Softmax Loss for Image Classification"
        https://arxiv.org/pdf/1805.03922.pdf
"""
import torch
from functools import partial
from itertools import combinations
from scipy.special import comb


def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n])
    I = torch.eye(n)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)


def rbf(X, sigma=None):
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))
    return hsic / (var1 * var2)


def linear_HSIC(X, Y):
    """
        Compute distance covariance
        :param X: 2d tensor
        :param Y: 2d tensor
        :return: scalar tensor
    """
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    """
        Normalized linear_HSIC score for independence.
        Equivalent to RV coefficient (Robert & Escoufier, 1976)
        and Tucker’s congruence coefficient (Tucker, 1951).
    :param X: 2d tensor
    :param Y: 2d tensor
    :return: scalar tensor
    """
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)


class CkaLoss:
    """
        Class for computing CKA loss - penalizing net pairwise similarity between
        nets of the same architecture. This loss promotes ensemble diversification.

        Register models on init with target layer names.
        Call after all models process a batch.
        proper use:
            > models = [MyModel() for x in range]
            > cka_batch_loss = CKA_loss(models, ['layer1.conv1'])
            > for batch in loader:
            >   for model in models:
            >       model(batch)
            >   loss = cka_batch_loss()

        The class supports Linear (Default) and RBG kernels.
        See "Similarity of Neural Network Representations Revisited" for kernel comparison.
    """
    def __init__(self, model_list, layer_name_list, rbf=False, sigma=0, layer_weights=None):
        assert len(model_list) > 1
        self.model_activations = dict()
        for i in range(len(model_list)):
            self.model_activations[i] = dict()

        self.layer_name_list = layer_name_list

        eq_weights = [1/len(self.layer_name_list)] * len(self.layer_name_list)
        self.weights = layer_weights if layer_weights is not None else eq_weights
        assert len(self.weights) == len(self.layer_name_list) and sum(self.weights) == 1

        def get_activation(name, index):
            def hook(model, input, output):
                output = output.detach().squeeze()
                if output.dim() > 2:  # in case of filters
                    output = output.flatten(1)
                self.model_activations[index][name] = output
            return hook

        # register layers for all models
        for i, model in enumerate(model_list):
            found = 0
            act_func = partial(get_activation, index=i)
            for name, module in model.named_modules():
                if name in self.layer_name_list:
                    module.register_forward_hook(act_func(name))
                    found += 1
            assert found == len(self.layer_name_list), 'not all layers are registered'

        self.cka_func = partial(kernel_CKA, sigma=sigma) if rbf else linear_CKA

    def __call__(self):
        """
            Here we collect selected layers and compute pairwise CKA loss.
            Loss between diff layers are equally weighted by default.
        """
        cka_loss = 0
        for layer_ind, name in enumerate(self.layer_name_list):
            activations = [self.model_activations[key][name] for key in self.model_activations]
            layer_loss = 0
            for i, j in combinations(range(len(self.model_activations)), 2):
                layer_loss += self.cka_func(activations[i], activations[j])
            layer_loss /= comb(len(self.model_activations), 2)
            cka_loss += self.weights[layer_ind]*layer_loss
        return cka_loss


if __name__ == '__main__':
    # maths sanity check
    X = torch.randn(100, 64)
    Y = torch.randn(100, 64)

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))
    print('Linear CKA, between Y and Y: {}'.format(linear_CKA(Y, Y)))
    print('')

    # CkaLoss check
    from torchvision import transforms as trans
    from torch.utils.data import DataLoader
    from torchvision import models, datasets

    res18_a = models.resnet18(pretrained=True)
    res18_b = models.resnet18(pretrained=False)
    cifar = datasets.CIFAR10('data')
    cifar.transform = trans.ToTensor()
    cifar_train_loader = DataLoader(cifar, batch_size=5)

    # loss == one
    model_list = [res18_a, res18_a]
    layers_names = ['layer4.1.conv2', 'layer4.0.conv2', 'layer3.1.conv2']

    cka = CkaLoss(model_list, layers_names)
    cka_rbf = CkaLoss(model_list, layers_names, rbf=True, sigma=10)

    for batch, label in cifar_train_loader: break
    for i,model in enumerate(model_list): model(batch)

    loss = cka()
    loss_rbf = cka_rbf()
    print('(net_a, net_a) Linear CKA loss is', loss.item())
    print('(net_a, net_a) RBF CKA loss is', loss_rbf.item())
    print('')

    # loss < one
    model_list = [res18_a, res18_b]

    cka = CkaLoss(model_list, layers_names)
    cka_rbf = CkaLoss(model_list, layers_names, rbf=True, sigma=10)

    for batch, label in cifar_train_loader: break
    for i,model in enumerate(model_list): model(batch)

    loss = cka()
    loss_rbf = cka_rbf()
    print('(net_a, net_b) Linear CKA loss is', loss.item())
    print('(net_a, net_b) RBF CKA loss is', loss_rbf.item())

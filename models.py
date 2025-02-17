from torchvision import models
from torch.nn import Conv2d, Linear, Sequential, Softmax, BatchNorm2d, Sigmoid, ReLU
from torch import nn
from torch.functional import F
from string import digits
import torch
from efficientnet_pytorch import EfficientNet

three_step_params = {'resnet18':[59, 44, -1], # 29, 14, 2
                     'resnet50':[158, 128, -1], # 71, 32, 2
                     'vgg16':[25, 13, -1],
                     'vgg19':[31, 23, 15, -1]}

def convert_syncbn_model(module, process_group=None):
    '''
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with :class:`apex.parallel.SyncBatchNorm`.
    All ``torch.nn.BatchNorm*N*d`` wrap around
    ``torch.nn.modules.batchnorm._BatchNorm``, so this function lets you easily switch
    to use sync BN.
    Args:
        module (torch.nn.Module): input module
    Example::
         # model is an instance of torch.nn.Module
    '''
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = torch.nn.SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine,
                                     module.track_running_stats, process_group)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_syncbn_model(child, process_group=process_group))
    del module
    return mod


class DenseRankHead(nn.Module):
    def __init__(self, init_net, cat=False, rank_out_features=None):
        super(DenseRankHead, self).__init__()
        self.cat = cat
        self.features = init_net.features
        self.classifier_label = init_net.classifier
        in_dim = init_net.classifier.in_features
        out_dim = rank_out_features or init_net.classifier.out_features
        rank_1 = Linear(in_features=in_dim if not cat else out_dim,
                        out_features=in_dim if not cat else out_dim, bias=True)

        rank_2 = Linear(in_features=in_dim if not cat else out_dim,
                        out_features=in_dim if not cat else out_dim, bias=True)

        rank_3 = Linear(in_features=in_dim if not cat else out_dim,
                        out_features=out_dim, bias=True)
        self.rank_classifier = Sequential(rank_1, ReLU(inplace=True), rank_2, ReLU(inplace=True), rank_3)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        label_out = torch.sigmoid(self.classifier_label(out))
        rank_input = label_out if self.cat else out
        rank_out = F.softmax(self.rank_classifier(rank_input), dim=1)
        return label_out, rank_out


class VGGRankHead(nn.Module):
    def __init__(self, init_net, cat=False, rank_out_features=None):
        super(VGGRankHead, self).__init__()
        self.cat = cat
        self.features = init_net.features
        self.avgpool = init_net.avgpool
        self.classifier_label = init_net.classifier
        rank_out_features = rank_out_features or init_net.classifier[-1].out_features
        self.rank_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, rank_out_features),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        label_out = torch.sigmoid(self.classifier_label(x))
        rank_input = label_out if self.cat else x
        rank_out = F.softmax(self.rank_classifier(rank_input), dim=1)
        return label_out, rank_out


class ResRankHead(nn.Module):
    def __init__(self, init_net, cat=False, rank_out_features=None):
        super(ResRankHead, self).__init__()
        components = ['conv1', 'bn1', 'relu', 'maxpool',
                      'layer1', 'layer2', 'layer3', 'layer4',
                      'avgpool', 'fc']
        for component in components:
            setattr(self, component, getattr(init_net, component))
        rank_out_features = rank_out_features or self.fc.out_featues
        self.rank_classifier = nn.Linear(self.fc.in_features, rank_out_features)
        self.cat = cat

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        label_out = torch.sigmoid(self.fc(x))
        rank_input = label_out if self.cat else x
        rank_out = F.softmax(self.rank_classifier(rank_input), dim=1)
        return label_out, rank_out

class PreBuildConverter:
    def __init__(self, in_channels, out_classes, add_rank=False, rank_out_features=None, cat=False, add_func=False, softmax=False, pretrained=False, half=False):
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.pretrained = pretrained
        self.half = half
        self.add_func = add_func
        self.add_rank = add_rank
        self.rank_out_features = rank_out_features
        self.softmax = softmax
        self.cat = cat

    def get_by_str(self, name):
        name_clean = name.translate(str.maketrans('', '', digits)).lower()
        ret_model = None
        if 'eff' in name_clean:
            # 'efficientnet-b0' ... 'efficientnet-b6'
            if self.pretrained:
                ret_model = EfficientNet.from_pretrained(name, num_classes=self.out_classes)
            else:
                ret_model = EfficientNet.from_name(name)
                # TODO add class support
        if 'vgg' in name_clean:
            ret_model = self.VGG(name)
            if self.add_rank:
                ret_model = VGGRankHead(ret_model, cat=self.cat, rank_out_features=self.rank_out_features)
        if 'dense' in name_clean:
            ret_model = self.DenseNet(name)
            if self.add_rank:
                ret_model = DenseRankHead(ret_model, cat=self.cat, rank_out_features=self.rank_out_features)
        if 'mobilenet' in name_clean:
            ret_model = self.MobileNet()
        if 'resnet' in name_clean:
            ret_model = self.ResNet(name)
            if self.add_rank:
                ret_model = ResRankHead(ret_model, cat=self.cat, rank_out_features=self.rank_out_features)
        if 'lenet' in name_clean:
            ret_model = self.LeNet(name)

        if self.add_func:
            ret_model = Sequential(ret_model, Softmax(1) if self.softmax else Sigmoid())

        if self.half:
            ret_model.half()  # convert to half precision
            for layer in ret_model.modules():
                if isinstance(layer, BatchNorm2d):
                    layer.float()

        return ret_model

    def VGG(self, name='vgg16'):
        model = getattr(models, name)(pretrained=self.pretrained)
        conv = model.features[0]
        classifier = model.classifier[-1]

        if conv.in_channels != self.in_channels:
            model.features[0] = Conv2d(self.in_channels, conv.out_channels,
                                       kernel_size=conv.kernel_size, stride=conv.stride,
                                       padding=conv.padding)
            model.features[0].bias.data = conv.bias
            model.features[0].weight.data = conv.weight.mean(1).unsqueeze(1)  # inherit og 1st layer weights

        if classifier.out_features != self.out_classes:
            model.classifier[-1] = Linear(in_features=classifier.in_features,
                                          out_features=self.out_classes, bias=True)

        return model

    def DenseNet(self, name='densenet121'):
        model = getattr(models, name)(pretrained=self.pretrained)
        conv = model.features[0]
        classifier = model.classifier

        if conv.in_channels != self.in_channels:
            model.features[0] = Conv2d(self.in_channels, conv.out_channels,
                                       kernel_size=conv.kernel_size, stride=conv.stride,
                                       padding=conv.padding, bias=conv.bias)
            model.features[0].bias.data = conv.bias
            model.features[0].weight.data = conv.weight.mean(1).unsqueeze(1)  # inherit og 1st layer weights

        if classifier.out_features != self.out_classes:
            model.classifier = Linear(in_features=classifier.in_features,
                                      out_features=self.out_classes, bias=True)

        return model

    def MobileNet(self):
        model = getattr(models, 'mobilenet_v2')(pretrained=self.pretrained)
        conv = model.features[0][0]
        classifier = model.classifier[-1]

        if conv.in_channels != self.in_channels:
            model.features[0][0] = Conv2d(self.in_channels, conv.out_channels,
                                          kernel_size=conv.kernel_size, stride=conv.stride,
                                          padding=conv.padding, bias=conv.bias)
            model.features[0][0].bias.data = conv.bias
            model.features[0][0].weight.data = conv.weight.mean(1).unsqueeze(1)  # inherit og 1st layer weights

        if classifier.out_features != self.out_classes:
            model.classifier[-1] = Linear(in_features=classifier.in_features,
                                          out_features=self.out_classes, bias=True)

        return model

    def ResNet(self, name='resnet50'):
        model = getattr(models, name)(pretrained=bool(self.pretrained))
        conv = model.conv1
        classifier = model.fc

        if conv.in_channels != self.in_channels:
            model.conv1 = Conv2d(self.in_channels, conv.out_channels,
                                          kernel_size=conv.kernel_size, stride=conv.stride,
                                          padding=conv.padding, bias=conv.bias)
            model.conv1.weight.data = conv.weight.mean(1).unsqueeze(1)  # inherit og 1st layer weights

        if classifier.out_features != self.out_classes:
            model.fc = Linear(in_features=classifier.in_features,
                                          out_features=self.out_classes, bias=True)

        return model


    def LeNet(self):
        model = getattr(models, 'GoogLeNet')(pretrained=self.pretrained)
        conv = model.conv1[0]
        classifier = model.fc

        if conv.in_channels != self.in_channels:
            model.conv1[0] = Conv2d(self.in_channels, conv.out_channels,
                                          kernel_size=conv.kernel_size, stride=conv.stride,
                                          padding=conv.padding, bias=conv.bias)
            model.conv1[0].bias.data = conv.bias
            model.conv1[0].weight.data = conv.weight.mean(1).unsqueeze(1)  # inherit og 1st layer weights

        if classifier.out_features != self.out_classes:
            model.fc = Linear(in_features=classifier.in_features,
                                          out_features=self.out_classes, bias=True)

        return model
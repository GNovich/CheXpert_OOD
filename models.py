from torchvision import models
from torch.nn import Conv2d, Linear, Sequential, Softmax, BatchNorm2d, Sigmoid
from string import digits
import torch

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

class PreBuildConverter:
    def __init__(self, in_channels, out_classes, add_func=False, softmax=False, pretrained=False, half=False):
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.pretrained = pretrained
        self.half = half
        self.add_func = add_func
        self.softmax = softmax

    def get_by_str(self, name):
        name_clean = name.translate(str.maketrans('', '', digits)).lower()
        ret_model = None
        if 'vgg' in name_clean:
            ret_model = self.VGG(name)
        if 'dense' in name_clean:
            ret_model = self.DenseNet(name)
        if 'mobilenet' in name_clean:
            ret_model = self.MobileNet()
        if 'resnet' in name_clean:
            ret_model = self.ResNet(name)
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
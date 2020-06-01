import torch
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as trans
from torchvision import datasets
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import roc_curve
import numpy as np
import os
from pathlib import Path

from models import PreBuildConverter, three_step_params
from utils import get_time, gen_plot, separate_bn_paras
from conf_table_TB import plot_confusion_matrix
plt.switch_backend('agg')

"""
    This is a simple CIFAR demonstration for CKA loss + pearson
"""
label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
# for resnet 0.layer4.0.conv2 0.layer3.1.conv2
# for densenet 0.features.denseblock3 0.features.denseblock4

class Learner(object):
    def __init__(self, conf, inference=False):

        # -----------   define model --------------- #
        self.n_classes = 10
        build_model = PreBuildConverter(in_channels=3, out_classes=self.n_classes, add_soft_max=True,
                                        pretrained=conf.pre_train)
        self.models = []
        for _ in range(conf.n_models):
            self.models.append(build_model.get_by_str(conf.net_mode).to(conf.device))
        print('{} {} models generated'.format(conf.n_models, conf.net_mode))

        # ------------  define params -------------- #
        if not inference:
            self.milestones = conf.milestones
            if not os.path.exists(conf.log_path):
                os.mkdir(conf.log_path)
            if not os.path.exists(conf.save_path):
                os.mkdir(conf.save_path)
            self.writer = SummaryWriter(logdir=conf.log_path)
            self.step = 0
            self.epoch = 0
            print('two model heads generated')

            self.get_opt(conf)
            print(self.optimizer)

        # ------------  define loaders -------------- #

        dloader_args = {
            'batch_size': conf.batch_size,
            'pin_memory': True,
            'num_workers': conf.num_workers,
            'drop_last': False,
        }
        self.ds = datasets.CIFAR10('data')
        self.ds.transform = trans.ToTensor()
        self.loader = DataLoader(self.ds, **dloader_args)
        eval_sampler = RandomSampler(self.ds, replacement=True, num_samples=len(self.ds) // 10)
        self.eval_loader = DataLoader(self.ds, sampler=eval_sampler, **dloader_args)

        self.cka_loss = conf.cka_loss(self.models, conf.cka_layers) if conf.cka else None
        if not inference:
            print('optimizers generated')
            self.running_loss = 0.
            self.running_pearson_loss = 0.
            self.running_ensemble_loss = 0.
            self.running_cka_loss = 0.

            self.board_loss_every = max(len(self.loader) // 4, 1)
            self.evaluate_every = conf.epoch_per_eval
            self.save_every = max(conf.epoch_per_save, 1)
            assert self.save_every >= self.evaluate_every

    def get_opt(self, conf):
        paras_only_bn = []
        paras_wo_bn = []
        for model in self.models:
            paras_only_bn_, paras_wo_bn_ = separate_bn_paras(model)
            paras_only_bn.append(paras_only_bn_)
            paras_wo_bn.append(paras_wo_bn_)

        self.optimizer = optim.SGD([
                                       {'params': paras_wo_bn[model_num], 'weight_decay': 5e-4}
                                       for model_num in range(conf.n_models)
                                   ] + [
                                       {'params': paras_only_bn[model_num]}
                                       for model_num in range(conf.n_models)
                                   ], lr=conf.lr, momentum=conf.momentum)

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        for mod_num in range(conf.n_models):
            torch.save(
                self.models[mod_num].state_dict(), Path(save_path) /
                                                   ('model_{}_{}_accuracy:{}_step:{}_{}.pth'.format(mod_num, get_time(),
                                                                                                    accuracy, self.step,
                                                                                                    extra)))
        torch.save(
            self.optimizer.state_dict(), Path(save_path) /
                                         ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                           self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path

        def load_fix(target_path):
            a = torch.load(target_path)
            fixed_a = {k.split('module.')[-1]: a[k] for k in a}
            torch.save(fixed_a, target_path)

        for mod_num in range(conf.n_models):
            target_path = save_path / 'model_{}_{}'.format(mod_num, fixed_str)
            load_fix(target_path)
            self.models[mod_num].load_state_dict(torch.load(target_path))
        if not model_only:
            for mod_num in range(conf.n_models):
                target_path = save_path / 'head_{}_{}'.format(mod_num, fixed_str)
                load_fix(target_path)
                self.heads[mod_num].load_state_dict(torch.load(target_path))
            target_path = save_path / 'optimizer_{}'.format(fixed_str)
            load_fix(target_path)
            self.optimizer.load_state_dict(torch.load(target_path))

    def board_val(self, db_name, accuracy, roc_curve_tensor, cm_fig):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        self.writer.add_figure('{}_conf_table'.format(db_name), cm_fig, self.step)

    def evaluate(self, conf, mode='test'):

        for i in range(len(self.models)):
            self.models[i].eval()

        do_mean = -1 if len(self.models) > 1 else 0
        ind_iter = range(do_mean, len(self.models))
        predictions = dict(zip(ind_iter, [[] for i in ind_iter]))
        prob = dict(zip(ind_iter, [[] for i in ind_iter]))
        labels = []
        pos = 2 if mode == 'train' else 1
        self.loader.dataset.train = False

        with torch.no_grad():
            for imgs, label in tqdm(self.eval_loader, total=len(self.eval_loader), desc=mode, position=pos):
                imgs = imgs.to(conf.device)

                self.optimizer.zero_grad()
                thetas = [model(imgs).detach() for model in self.models]
                if len(self.models) > 1: thetas = [torch.mean(torch.stack(thetas), 0)] + thetas
                for ind, theta in zip(range(do_mean, len(self.models)), thetas):
                    val, arg = torch.max(theta, dim=1)
                    predictions[ind].append(arg.cpu().numpy())
                    prob[ind].append(theta.cpu().numpy())
                labels.append(label.detach().cpu().numpy())

        labels = np.hstack(labels)
        results = []
        for ind in range(do_mean, len(self.models)):
            curr_predictions = np.hstack(predictions[ind])
            curr_prob = np.vstack(prob[ind])

            # Compute ROC curve and ROC area for each class
            img_d_fig = plot_confusion_matrix(labels, curr_predictions, label_names, tensor_name='dev/cm_' + mode)
            res = (curr_predictions == labels)
            acc = sum(res) / len(res)
            fpr, tpr, _ = roc_curve(np.repeat(res, self.n_classes), curr_prob.ravel())
            buf = gen_plot(fpr, tpr)
            roc_curve_im = Image.open(buf)
            roc_curve_tensor = trans.ToTensor()(roc_curve_im)
            results.append((acc, roc_curve_tensor, img_d_fig))
        return results

    def pretrain(self, conf):
        for model_num in range(conf.n_models):
            self.models[model_num].train()
            if not conf.cpu_mode:
                device_ids = list(range(conf.ngpu))
                self.models[model_num] = torch.nn.DataParallel(self.models[model_num], device_ids=device_ids)
            self.models[model_num].to(conf.device)

        pre_layers = three_step_params[conf.net_mode] if len(conf.pre_layers) < 1 else conf.pre_layers
        assert len(pre_layers) == len(conf.pre_steps)
        last_stage = len(pre_layers) - 1
        for stage, (layer_step, layer_epoch) in enumerate(zip(pre_layers, conf.pre_steps)):
            for model_num in range(conf.n_models):
                for i, (name, param) in enumerate(self.models[model_num].named_parameters()):
                    param.requires_grad = (i > layer_step) or ('bn' in name)
            self.train(conf, layer_epoch, save_final=(stage == last_stage))

    def train(self, conf, epochs, save_final=True):
        if not conf.pre_train:
            for model_num in range(conf.n_models):
                self.models[model_num].train()
                if not conf.cpu_mode:
                    device_ids = list(range(conf.ngpu))
                    self.models[model_num] = torch.nn.DataParallel(self.models[model_num], device_ids=device_ids)
                self.models[model_num].to(conf.device)
            self.running_loss = 0.
            self.running_pearson_loss = 0.
            self.running_ensemble_loss = 0.

        epoch_iter = range(epochs)
        accuracy = 0
        for e in epoch_iter:
            # check lr update
            for milestone in self.milestones:
                if self.epoch == milestone:
                    self.schedule_lr()

            # train
            self.loader.dataset.train = True
            for imgs, labels in tqdm(self.loader, desc='epoch {}'.format(e), total=len(self.loader), position=0):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)

                self.optimizer.zero_grad()

                # calc embeddings
                thetas = []
                joint_losses = []
                for model_num in range(conf.n_models):
                    theta = self.models[model_num](imgs)
                    thetas.append(theta)
                    joint_losses.append(conf.ce_loss(theta, labels))
                joint_losses = sum(joint_losses) / max(len(joint_losses), 1)

                # calc loss
                if conf.pearson:
                    outputs = torch.stack(thetas)
                    pearson_corr_models_loss = conf.pearson_loss(outputs, labels)
                    self.running_pearson_loss += pearson_corr_models_loss.item()
                    if conf.cka:
                        pearson_corr_models_loss += self.cka_loss()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * pearson_corr_models_loss
                elif conf.joint_mean:
                    mean_output = torch.mean(torch.stack(thetas), 0)
                    ensemble_loss = conf.ce_loss(mean_output, labels)
                    self.running_ensemble_loss += ensemble_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * ensemble_loss
                elif conf.cka:
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * self.cka_loss()
                else:
                    loss = joint_losses

                loss.backward()
                self.running_loss += loss.item()
                self.optimizer.step()

                # listen to running losses
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = self.running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    self.running_loss = 0.
                    if conf.pearson:  # ganovich listening to pearson
                        loss_board = self.running_pearson_loss / self.board_loss_every
                        self.writer.add_scalar('pearson_loss', loss_board, self.step)
                        self.running_pearson_loss = 0.

                    if conf.joint_mean:
                        loss_board = self.running_ensemble_loss / self.board_loss_every
                        self.writer.add_scalar('ensemble_loss', loss_board, self.step)
                        self.running_ensemble_loss = 0.

                self.step += 1

            # listen to validation and save every so often
            if self.epoch % self.evaluate_every == 0:# and self.epoch != 0:
                for mode in ['test', 'train']:
                    results = self.evaluate(conf=conf, mode=mode)
                    do_mean = -1 if len(self.models) > 1 else 0
                    for model_num, (accuracy, roc_curve_tensor, img_d_fig) in zip(range(do_mean, conf.n_models), results):
                        broad_name = 'mod_'+mode+'_' + str(model_num if model_num>-1 else 'mean')
                        self.board_val(broad_name, accuracy, roc_curve_tensor, img_d_fig)
                        if model_num > -1: self.models[model_num].train()

            if self.epoch % self.save_every == 0 and self.epoch != 0:
                self.save_state(conf, accuracy)
            self.epoch += 1

        if accuracy is not None and save_final:
            self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)


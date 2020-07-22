import torch
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from scipy.stats import entropy
from PIL import Image
from torchvision import transforms as trans
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score
from sklearn.utils import compute_sample_weight
import numpy as np
import os
from pathlib import Path
from Datasets import get_isic
from models import PreBuildConverter, three_step_params
from utils import get_time, gen_plot_mult, separate_bn_paras, StratifiedSampler
from conf_table_TB import plot_confusion_matrix
plt.switch_backend('agg')


class Learner(object):
    def __init__(self, conf, inference=False):
        # ------------  define dataset -------------- #
        dat_args = {'ood': conf.ood,
                    'with_rank': conf.rank
                    }
        self.ds_train, self.ds_test = get_isic(**dat_args)
        self.n_classes = len(self.ds_train.classes)

        # ------------  define loaders -------------- #
        dloader_args = {
            'batch_size': conf.batch_size,
            'pin_memory': True,
            'num_workers': conf.num_workers,
            'drop_last': False,
        }
        self.loader = DataLoader(self.ds_train, sampler=StratifiedSampler(self.ds_train), **dloader_args)

        eval_sampler = RandomSampler(self.ds_test, replacement=True, num_samples=len(self.ds_test) // 2)
        dloader_args = {
            'batch_size': int(np.ceil(conf.batch_size / 5)),
            'pin_memory': False,
            'num_workers': conf.num_workers,
            'drop_last': False,
        }
        self.eval_loader = DataLoader(self.ds_test, sampler=eval_sampler, **dloader_args)
        # self.eval_loader = DataLoader(self.ds_test, **dloader_args)

        train_eval_sampler = RandomSampler(self.ds_train, replacement=True, num_samples=len(self.ds_train) // 10)
        dloader_args = {
            'batch_size': int(np.ceil(conf.batch_size / 5)),
            'pin_memory': False,
            'num_workers': conf.num_workers,
            'drop_last': False,
        }
        self.train_eval_loader = DataLoader(self.ds_train, sampler=train_eval_sampler, **dloader_args)

        # -----------   define model --------------- #
        self.n_classes = len(self.ds_train.classes)
        build_model = PreBuildConverter(in_channels=3, out_classes=self.n_classes, add_func=True, softmax=True,
                                        pretrained=conf.pre_train)
        self.models = []
        for _ in range(conf.n_models):
            self.models.append(build_model.get_by_str(conf.net_mode).to(conf.device))
        print('{} {} models generated'.format(conf.n_models, conf.net_mode))

        # ------------  define params -------------- #
        if not inference:
            # rebalance loss
            conf.ce_loss = CrossEntropyLoss(weight=self.ds_train.class_weights.to(conf.device))

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
            #self.scheduler = StepLR(self.optimizer, step_size=25, gamma=0.2)

            print('optimizers generated')
            self.running_loss = 0.
            self.running_pearson_loss = 0.
            self.running_ensemble_loss = 0.
            self.running_cka_loss = 0.

            self.board_loss_every = max(len(self.loader) // 2, 1)
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

        self.optimizer = optim.Adam([
                                       {'params': paras_wo_bn[model_num], 'weight_decay': 5e-4}
                                       for model_num in range(conf.n_models)
                                   ] + [
                                       {'params': paras_only_bn[model_num]}
                                       for model_num in range(conf.n_models)
                                   ], lr=conf.lr)

        """
        self.optimizer = optim.SGD([
                                       {'params': paras_wo_bn[model_num], 'weight_decay': 5e-4}
                                       for model_num in range(conf.n_models)
                                   ] + [
                                       {'params': paras_only_bn[model_num]}
                                       for model_num in range(conf.n_models)
                                   ], lr=conf.lr, momentum=conf.momentum)
        """
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
        self.writer.add_scalar('{}_recall'.format(db_name), accuracy, self.step)
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

        eval_loader = self.train_eval_loader if mode == 'train' else self.eval_loader
        has_ood = eval_loader.dataset.ood
        report_ood = has_ood and (do_mean == -1) and mode != 'train'

        eval_loader.dataset.evaluate()
        with torch.no_grad():
            for imgs, label in tqdm(eval_loader, total=len(eval_loader), desc=mode, position=pos):
                imgs = imgs.to(conf.device)
                bs, n_crops, c, h, w = imgs.size()
                imgs = imgs.view(-1, c, h, w).cuda()

                self.optimizer.zero_grad()
                thetas = []
                for model_num in range(conf.n_models):
                    theta = self.models[model_num](imgs)
                    theta = theta.view(bs, n_crops, -1).mean(1).detach()
                    thetas.append(theta.detach())

                if len(self.models) > 1: thetas = [torch.mean(torch.stack(thetas), 0)] + thetas
                for ind, theta in zip(range(do_mean, len(self.models)), thetas):
                    val, arg = torch.max(theta, dim=1)
                    predictions[ind].append(arg.cpu().numpy())
                    prob[ind].append(theta.cpu().numpy())
                labels.append(label.detach().cpu().numpy())

        labels = np.hstack(labels)
        predictions = {key: np.hstack(predictions[key]) for key in predictions}
        prob = {key: np.vstack(prob[key]) for key in prob}
        results = []
        label_names = eval_loader.dataset.classes
        class_weight = dict(enumerate(self.ds_train.class_weights.numpy()))
        for ind in range(do_mean, len(self.models)):
            curr_predictions = predictions[ind]
            curr_prob = prob[ind]
            eval_labels = labels
            if has_ood:
                # ood filtering requeired
                ood_ind = has_ood
                is_ood = labels != ood_ind
                eval_labels = labels[is_ood]
                curr_predictions = curr_predictions[labels != ood_ind]
                curr_prob = curr_prob[labels != ood_ind]

            # Compute ROC curve and ROC area for each class
            img_d_fig = plot_confusion_matrix(eval_labels, curr_predictions, label_names,
                                              tensor_name='dev/cm_' + mode)

            # sample_w = compute_sample_weight(class_weight, eval_labels)
            recall = recall_score(eval_labels, curr_predictions, average='weighted')
            # res = (curr_predictions == eval_labels)
            # acc = sum(res) / len(res)

            dummies = np.eye(self.n_classes)[eval_labels]
            fpr = dict()
            tpr = dict()
            for i in range(self.n_classes):
                fpr[i], tpr[i], _ = roc_curve(dummies[:, i], curr_prob[:, i])

            if report_ood and ind == do_mean:
                # ood eval
                ood_ind = has_ood
                ensemble_prob = np.stack([prob[ind] for ind in range(len(self.models))]).mean(0)
                #ensemble_pred = np.stack([predictions[ind] for ind in range(len(self.models))]).T
                ood_confidance = entropy(ensemble_prob, axis=1, base=ensemble_prob.shape[1])
                fpr[ood_ind], tpr[ood_ind], _ = roc_curve(is_ood, ood_confidance)
                ood_auc = roc_auc_score(is_ood, ood_confidance)
                roc_labels = label_names
            else:
                roc_labels = label_names if mode == 'train' else label_names[:-1]

            buf = gen_plot_mult(fpr, tpr, roc_labels)
            roc_curve_im = Image.open(buf)
            roc_curve_tensor = trans.ToTensor()(roc_curve_im)
            results.append((recall, roc_curve_tensor, img_d_fig))

        return results if (not report_ood) else (results, ood_auc)

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
            self.running_cka_loss = 0.
            self.running_ncl_loss = 0.

        epoch_iter = range(epochs)
        accuracy = 0
        for e in epoch_iter:
            # check lr update
            for milestone in self.milestones:
                if self.epoch == milestone:
                    self.schedule_lr()

            # train
            self.loader.dataset.train()
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
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * pearson_corr_models_loss
                elif conf.joint_mean:
                    mean_output = torch.mean(torch.stack(thetas), 0)
                    ensemble_loss = conf.ce_loss(mean_output, labels)
                    self.running_ensemble_loss += ensemble_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * ensemble_loss
                elif conf.ncl:
                    outputs = torch.stack(thetas)
                    alpha = conf.alpha
                    ncl_loss = conf.ncl_loss(outputs, labels)
                    self.running_ncl_loss += ncl_loss.item()
                    loss = (1 - alpha) * joint_losses + alpha * ncl_loss
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

                    if conf.cka:
                        loss_board = self.running_cka_loss / self.board_loss_every
                        self.writer.add_scalar('cka_loss', loss_board, self.step)
                        self.running_cka_loss = 0.

                    if conf.ncl:
                        loss_board = self.running_cka_loss / self.board_loss_every
                        self.writer.add_scalar('ncl_loss', loss_board, self.step)
                        self.running_cka_loss = 0.

                self.step += 1

            # listen to validation and save every so often
            if self.epoch % self.evaluate_every == 0: # and self.epoch != 0:
                for mode in ['test', 'train']:
                    results = self.evaluate(conf=conf, mode=mode)
                    if mode != 'train' and conf.ood and len(self.models) > 1:
                        results, ood_auc = results
                        self.writer.add_scalar('ood_auc', ood_auc, self.step)

                    do_mean = -1 if len(self.models) > 1 else 0
                    for model_num, (recall, roc_curve_tensor, img_d_fig) in zip(range(do_mean, conf.n_models), results):
                        broad_name = 'mod_'+mode+'_' + str(model_num if model_num>-1 else 'mean')
                        self.board_val(broad_name, recall, roc_curve_tensor, img_d_fig)
                        if model_num > -1: self.models[model_num].train()

            if self.epoch % self.save_every == 0 and self.epoch != 0:
                self.save_state(conf, accuracy)
            self.epoch += 1
            #self.scheduler.step()

        if accuracy is not None and save_final:
            self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 2
        print(self.optimizer)


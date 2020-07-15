"""
    Gal Novich skeleton learner
    follows parametrization of for pretrained densenet121
    NIH https://github.com/zoogzog/chexnet
    CheXpert https://github.com/jfhealthcare/Chexpert/

"""
import torch
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from PIL import Image
from torchvision import transforms as trans
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from models import PreBuildConverter, three_step_params
from utils import get_time, separate_bn_paras, gen_plot
from conf_table_TB import plot_auc_vector, plot_confusion_matrix
from Datasets import get_nih, get_chexpert

plt.switch_backend('agg')


class Learner(object):
    def __init__(self, conf, inference=False):

        # ------------  define dataset -------------- #
        if conf.dat_mode == 'nih':
            self.ds_train, self.ds_test = get_nih()
        elif conf.dat_mode.lower() == 'chexpert':
            chexpert_args = {'No_finding': conf.use_clean,
                             'parenchymal': conf.use_int,
                             'extraparenchymal': conf.use_ext,
                             'limit_out_labels': conf.ood_limit,
                             'with_rank': conf.rank
                             }
            self.ds_train, self.ds_test, self.out_in_ds, self.out_out_ds = get_chexpert(**chexpert_args)
            self.ds_morph = get_chexpert(morph_load=conf.morph, **chexpert_args)
        else:
            raise ValueError('no such dataset')

        self.n_classes = len(self.ds_train.label_names)

        # ------------  define loaders -------------- #
        dloader_args = {
            'batch_size': conf.batch_size,
            'pin_memory': True,
            'num_workers': conf.num_workers,
            'drop_last': False,
        }
        self.loader = DataLoader(self.ds_train, **dloader_args)
        eval_sampler = RandomSampler(self.ds_test, replacement=True, num_samples=len(self.ds_test) // 10)
        self.eval_loader = DataLoader(self.ds_test, sampler=eval_sampler, **dloader_args)
        self.morph_loader = DataLoader(self.ds_morph, **dloader_args)

        # -----------   define models --------------- #
        build_model = PreBuildConverter(in_channels=3, out_classes=self.n_classes,
                                        add_rank=conf.rank, #add_func=True, softmax=False,  # sigmoid
                                        rank_out_features=None if not conf.rank else self.ds_train.n_rank_labels,
                                        pretrained=conf.pre_train, cat=conf.cat)
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
            if conf.dat_mode.lower() == 'chexpert':
                tables = [self.ds_train.table, self.ds_test.table,
                          None if self.out_in_ds is None else self.out_in_ds.table , self.out_out_ds.table]
                names = ['ds_train', 'ds_test', 'out_in', 'out_out']
                for name, table in zip(names, tables):
                    if table is None:
                        continue
                    table.to_csv(os.path.join(conf.save_path, name))

            self.step = 0
            self.epoch = 0

            self.get_opt(conf)
            print(self.optimizer)
            print('optimizers generated')

            # ------------  define loss -------------- #
            self.cka_loss = conf.cka_loss(self.models, conf.cka_layers) if conf.cka else None
            self.running_loss = 0.
            self.running_pearson_loss = 0.
            self.running_ensemble_loss = 0.
            self.running_cka_loss = 0.
            self.running_ncl_loss = 0.
            self.running_morph_loss = 0.
            self.running_rank_loss = 0.
            self.running_rank_pearson_loss = 0.

            # ------------  define save/log times -------------- #
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

        param_list = [
                         {'params': paras_wo_bn[model_num], 'weight_decay': 5e-4}
                         for model_num in range(conf.n_models)
                     ] + [
                         {'params': paras_only_bn[model_num]}
                         for model_num in range(conf.n_models)
                     ]

        self.optimizer = optim.Adam(param_list, lr=conf.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')

    def save_state(self, conf, auc, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        for mod_num in range(conf.n_models):
            torch.save(self.models[mod_num].state_dict(),
                       Path(save_path) / ('model_{}_{}_auc:{}_step:{}_{}.pth'.format(
                           mod_num, get_time(), auc, self.step, extra)))
        torch.save(self.optimizer.state_dict(),
                   Path(save_path) / ('optimizer_{}_auc:{}_step:{}_{}.pth'.format(
                       get_time(), auc, self.step, extra)))

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
            target_path = save_path / 'optimizer_{}'.format(fixed_str)
            load_fix(target_path)
            self.optimizer.load_state_dict(torch.load(target_path))

    def board_val(self, db_name, auc, auc_fig):
        self.writer.add_scalar('{}_auc'.format(db_name), auc, self.step)
        self.writer.add_figure('{}_auc_vec'.format(db_name), auc_fig, self.step)

    def board_val_rank(self, db_name, accuracy, roc_curve_tensor, cm_fig):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        self.writer.add_figure('{}_conf_table'.format(db_name), cm_fig, self.step)

    def evaluate(self, conf, mode='test'):
        for i in range(len(self.models)):
            self.models[i].eval()

        do_mean = -1 if len(self.models) > 1 else 0
        ind_iter = range(do_mean, len(self.models))
        prob = dict(zip(ind_iter, [[] for i in ind_iter]))
        rank_prob = dict(zip(ind_iter, [[] for i in ind_iter]))
        rank_predictions = dict(zip(ind_iter, [[] for i in ind_iter]))
        labels = []
        rank_labels = []
        pos = 2 if mode == 'train' else 1
        self.loader.dataset.train = False
        with torch.no_grad():
            for imgs, label in tqdm(self.eval_loader, total=len(self.eval_loader), desc=mode, position=pos):
                imgs = imgs.to(conf.device)
                if conf.rank:
                    label, rank_label = label
                    rank_labels.append(rank_label.detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

                #bs, n_crops, c, h, w = imgs.size()
                #imgs = imgs.view(-1, c, h, w).cuda()

                self.optimizer.zero_grad()
                #thetas = [model(imgs).view(bs, n_crops, -1).mean(1).detach() for model in self.models]
                thetas = []
                rank_thetas = []
                for model_num in range(conf.n_models):
                    if conf.rank:
                        theta, rank_theta = self.models[model_num](imgs)
                        rank_thetas.append(rank_theta.detach())
                    else:
                        theta = self.models[model_num](imgs)
                    thetas.append(theta.detach())

                if len(self.models) > 1: thetas = [torch.mean(torch.stack(thetas), 0)] + thetas
                for ind, theta in zip(range(do_mean, len(self.models)), thetas):
                    prob[ind].append(theta.cpu().numpy())

                if conf.rank:
                    if len(self.models) > 1: rank_thetas = [torch.mean(torch.stack(rank_thetas), 0)] + rank_thetas
                    for ind, theta in zip(range(do_mean, len(self.models)), rank_thetas):
                        val, arg = torch.max(theta, dim=1)
                        rank_predictions[ind].append(arg.cpu().numpy())
                        rank_prob[ind].append(theta.cpu().numpy())

        labels = np.vstack(labels)
        if conf.rank:
            rank_labels = np.hstack(rank_labels)
        results = []
        for ind in range(do_mean, len(self.models)):
            cur_res = []
            curr_prob = np.vstack(prob[ind])

            AUROCs = []
            for i in range(self.n_classes):
                AUROCs.append(roc_auc_score(labels[:, i], curr_prob[:, i]))
            AUROC_avg = np.array(AUROCs).mean()
            img_d_fig = plot_auc_vector(AUROCs, self.ds_test.label_names)
            cur_res.append((AUROC_avg, img_d_fig))

            if conf.rank:
                curr_predictions = np.hstack(rank_predictions[ind])
                curr_prob = np.vstack(rank_prob[ind])

                img_d_fig = plot_confusion_matrix(rank_labels, curr_predictions, self.ds_test.label_names, tensor_name='dev/cm_' + mode)
                res = (curr_predictions == rank_labels)
                acc = sum(res) / len(res)
                fpr, tpr, _ = roc_curve(np.repeat(res, self.ds_test.n_rank_labels), curr_prob.ravel())
                buf = gen_plot(fpr, tpr)
                roc_curve_im = Image.open(buf)
                roc_curve_tensor = trans.ToTensor()(roc_curve_im)
                cur_res.append((acc, roc_curve_tensor, img_d_fig))

            results.append(cur_res)
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
            self.running_cka_loss = 0.
            self.running_ncl_loss = 0.
            self.running_morph_loss = 0.
            self.running_rank_loss = 0.
            self.running_rank_pearson_loss = 0.

        epoch_iter = range(epochs)
        accuracy = 0

        for e in epoch_iter:
            # check lr update
            for milestone in self.milestones:
                if self.epoch == milestone:
                    self.schedule_lr()

            # train
            self.loader.dataset.train = True
            morph_iter = iter(self.morph_loader)
            for imgs, labels in tqdm(self.loader, desc='epoch {}'.format(e), total=len(self.loader), position=0):
                imgs = imgs.to(conf.device)

                if conf.rank:
                    labels, rank_labels = labels
                    rank_labels = rank_labels.to(conf.device)
                labels = labels.to(conf.device)

                if conf.morph:
                    try:
                        morph_imgs, morph_labels = next(morph_iter)
                    except StopIteration:
                        morph_iter = iter(self.morph_loader)
                        morph_imgs, morph_labels = next(morph_iter)
                    morph_imgs = morph_imgs.to(conf.device)
                    morph_labels = morph_labels.to(conf.device)

                self.optimizer.zero_grad()

                # calc embeddings
                thetas = []
                rank_thetas = []
                joint_losses = []
                for model_num in range(conf.n_models):
                    if conf.rank:
                        theta, rank_theta = self.models[model_num](imgs)
                        rank_thetas.append(rank_theta)
                    else:
                        theta = self.models[model_num](imgs)
                    thetas.append(theta)
                    joint_losses.append(conf.ce_loss(theta, labels))
                joint_losses = sum(joint_losses) / max(len(joint_losses), 1)


                if conf.morph:
                    morph_loss = []
                    for model_num in range(conf.n_models):
                        eta_hat = self.models[model_num](morph_imgs)
                        if conf.morph_target == 'zero':
                            correct_values = (eta_hat * morph_labels).sum(-1) / morph_labels.sum(-1)
                            zero_labels = torch.zeros_like(correct_values)
                            morph_loss.append(conf.morph_loss(correct_values, zero_labels))
                        else:
                            correct_values = (eta_hat * morph_labels).sum(-1) / morph_labels.sum(-1)
                            average_values = eta_hat.mean(-1)
                            morph_loss.append(conf.morph_loss(correct_values, average_values))
                    morph_loss = sum(morph_loss) / max(len(morph_loss), 1)
                    morph_loss *= conf.morph_alpha
                    self.running_morph_loss += morph_loss.item()
                    joint_losses = joint_losses + morph_loss


                if conf.pearson:
                    outputs = torch.stack(thetas)
                    # we are ignoring the 'No Finding' label here.
                    if conf.use_clean:
                        not_clean = labels[:, 0] != 1
                        pearson_corr_models_loss = conf.pearson_loss(outputs[:, not_clean, 1:], labels[not_clean, 1:])
                    else:
                        pearson_corr_models_loss = conf.pearson_loss(outputs, labels)
                    self.running_pearson_loss += pearson_corr_models_loss.item()
                    if conf.cka:
                        cka_loss = self.cka_loss()
                        self.running_cka_loss += cka_loss.item()
                        pearson_corr_models_loss += cka_loss
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * pearson_corr_models_loss
                elif conf.joint_mean:
                    mean_output = torch.prod(torch.stack(thetas), 0)
                    ensemble_loss = conf.ce_loss(mean_output, labels)
                    self.running_ensemble_loss += ensemble_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * ensemble_loss
                elif conf.cka:
                    alpha = conf.alpha
                    cka_loss = self.cka_loss()
                    self.running_cka_loss += cka_loss.item()
                    loss = (1 - alpha) * joint_losses + alpha * cka_loss
                elif conf.ncl:
                    outputs = torch.stack(thetas)
                    alpha = conf.alpha
                    ncl_loss = conf.ncl_loss(outputs, labels)
                    self.running_ncl_loss += ncl_loss.item()
                    loss = (1 - alpha) * joint_losses + alpha * ncl_loss
                else:
                    loss = joint_losses

                # rank loss segment
                if conf.rank:
                    rank_losses = []
                    for theta in rank_thetas:
                        rank_losses.append(conf.rank_loss(theta, rank_labels))
                    rank_loss = sum(rank_losses) / max(len(rank_losses), 1)
                    self.running_rank_loss += rank_loss.item()

                    if conf.rank_pearson:
                        # eneter label rank convertere here?
                        pearson_rank_loss = conf.rank_pearson_loss(torch.stack(rank_thetas), rank_labels)
                        self.running_rank_pearson_loss += pearson_rank_loss.item()
                        alpha = conf.rank_pearson_alpha
                        rank_loss = (1 - alpha) * rank_loss + alpha * pearson_rank_loss

                    alpha = conf.rank_alpha
                    loss = (1 - alpha) * loss + alpha * rank_loss


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
                        loss_board = self.running_ncl_loss / self.board_loss_every
                        self.writer.add_scalar('ncl_loss', loss_board, self.step)
                        self.running_ncl_loss = 0.

                    if conf.morph:
                        loss_board = self.running_morph_loss / self.board_loss_every
                        self.writer.add_scalar('morph_loss', loss_board, self.step)
                        self.running_morph_loss = 0.

                    if conf.rank:
                        loss_board = self.running_rank_loss / self.board_loss_every
                        self.writer.add_scalar('rank_loss', loss_board, self.step)
                        self.running_rank_loss = 0.
                        if conf.rank_pearson:
                            loss_board = self.running_rank_pearson_loss / self.board_loss_every
                            self.writer.add_scalar('rank_pearson_loss', loss_board, self.step)
                            self.running_rank_pearson_loss = 0.

                self.step += 1

            self.scheduler.step(loss.data)
            # listen to validation and save every so often
            if self.epoch % self.evaluate_every == 0:  # and self.epoch != 0:
                for mode in ['test', 'train']:
                    results = self.evaluate(conf=conf, mode=mode)
                    do_mean = -1 if len(self.models) > 1 else 0
                    for cur_res in zip(range(do_mean, conf.n_models), results):
                        if conf.rank:
                            model_num, (label_res, rank_res) = cur_res
                            acc, roc_curve_tensor, rank_img_d_fig = rank_res
                            broad_name = 'mod_' + mode + '_' + str(model_num if model_num > -1 else 'mean')
                            self.board_val_rank(broad_name, acc, roc_curve_tensor, rank_img_d_fig)
                            auc, img_d_fig = label_res
                        else:
                            model_num, label_res = cur_res
                            auc, img_d_fig = label_res

                        broad_name = 'mod_' + mode + '_' + str(model_num if model_num > -1 else 'mean')
                        self.board_val(broad_name, auc, img_d_fig)
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

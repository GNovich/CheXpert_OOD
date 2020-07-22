from config import get_config
from Learner_ISIC import Learner
import argparse
import torch
from functools import partial
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from Pearson import ncl_loss, pearson_corr_loss_multilabel, pearson_corr_loss
from CKA_torch import CkaLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for CBIS-DDSM')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="number of workers", default=5, type=int)
    parser.add_argument("-s", "--epoch_per_save", help="save_every s epochs", default=50, type=int)
    parser.add_argument("-eval", "--epoch_per_eval", help="eval_every eval epochs", default=5, type=int)
    parser.add_argument("-net", "--net_mode", help="choose net", default='resnet50', type=str)
    parser.add_argument("-dat", "--dat_mode", help="choose dataset", default='nih', type=str)

    parser.add_argument("-n", "--n_models", help="how many duplicate nets to use. 1 leads to basic training, "
                                                 "making -a and -p flags redundant", default=1, type=int)

    parser.add_argument("-pre", "--pre_train", help="use a pretrain net?", default=0, type=int)
    parser.add_argument("-pre_layers", "--pre_layers", help="layer steps to use?", default=[], type=int, nargs='*')
    parser.add_argument("-pre_step", "--pre_steps", help="what steps to use?", default=[], type=int, nargs='*')
    parser.add_argument("-ngpu", "--ngpu", help="how many gpu's to use?", default=None, type=int)

    parser.add_argument("-m", "--milestones", help="fractions of where lr will be tuned", default=[], type=int, nargs='*')
    parser.add_argument("-a", "--alpha", help="balancing parameter", default=0, type=float)
    parser.add_argument("-t", "--sig_thresh", help="thresholding of the most correct class", default=0.9, type=float)
    parser.add_argument("-p", "--pearson", help="using pearson loss", default=False, type=bool)
    parser.add_argument("-cka", "--cka", help="using cka loss", default=False, type=bool)
    parser.add_argument("-cka_layers", "--cka_layers", help="using cka loss", default=[], type=str, nargs='*')
    parser.add_argument("-ncl", "--ncl", help="using Negative Correlation Loss", default=False, type=bool)
    parser.add_argument("-mean", "--joint_mean", help="using mean loss", default=False, type=bool)
    parser.add_argument("-morph", "--morph", help="use a morph", default=False, type=int)
    parser.add_argument("-morph_a", "--morph_alpha", help="balance parameter", default=10., type=float)
    parser.add_argument("-morph_t", "--morph_target", help="chosse morph tactic", default='zero', type=str)
    parser.add_argument("-morph_dumb", "--morph_dumb", help="use dumb morphs", default=0, type=int)

    # rank
    parser.add_argument("-rank", "--rank", help="use rank loss", default=0, type=int)
    parser.add_argument("-rank_a", "--rank_alpha", help="balance with other loss componenet", default=0, type=float)
    parser.add_argument("-rank_p", "--rank_pearson", help="use rank pearson", default=0, type=int)
    parser.add_argument("-rank_p_a", "--rank_pearson_alpha", help="balance within rank loss", default=0, type=float)
    parser.add_argument("-cat", "--cat", help="make learner a concat or slide", default=0, type=int)

    # xray exp
    parser.add_argument("-use_clean", "--clean_label", help="use 'No Finding; class", default=1, type=int)
    parser.add_argument("-use_int", "--parenchymal", help="use parenchymal classes", default=1, type=int)
    parser.add_argument("-use_ext", "--extraparenchymal", help="use extraparenchymal classes", default=1, type=int)
    parser.add_argument("-ood_limit", "--ood_limit", help="exlude all a positive samples of an ood class", default=0, type=int)

    parser.add_argument("-logdir", "--logdir", help="extend log/saves to a folder for group experiment", default='', type=str)

    parser.add_argument("-c", "--cpu_mode", help="force cpu mode", default=0, type=int)

    # leison
    parser.add_argument("-ood", "--ood_label", help="choose ood label", default=None, type=str)
    parser.add_argument("-valid", "--valid_size", help="choose validation ratio", default=0, type=str)

    args = parser.parse_args()
    conf = get_config(logext=args.logdir)

    # training param
    conf.dat_mode = args.dat_mode
    conf.ngpu = args.ngpu or torch.cuda.device_count()
    conf.pre_layers = args.pre_layers
    conf.pre_steps = args.pre_steps
    conf.pre_train = args.pre_train
    conf.net_mode = args.net_mode
    conf.epoch_per_eval = args.epoch_per_eval
    conf.epoch_per_save = args.epoch_per_save
    conf.cpu_mode = args.cpu_mode
    conf.device = torch.device("cuda" if (torch.cuda.is_available() and not conf.cpu_mode) else "cpu")
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.epochs = args.epochs
    conf.milestones = args.milestones

    # pearson param
    conf.alpha = args.alpha
    conf.sig_thresh = args.sig_thresh
    conf.n_models = args.n_models
    conf.pearson = args.pearson
    conf.cka = args.cka
    conf.cka_layers = args.cka_layers
    conf.joint_mean = args.joint_mean
    conf.ncl = args.ncl

    # rank param
    conf.rank = args.rank
    conf.rank_alpha = args.rank_alpha
    conf.rank_pearson = args.rank_pearson
    conf.rank_pearson_alpha = args.rank_pearson_alpha
    conf.cat = args.cat

    # morph param
    conf.morph_alpha = args.morph_alpha
    conf.morph = args.morph
    conf.morph_target = args.morph_target
    conf.morph_dumb = args.morph_dumb

    # loss funcs
    conf.ce_loss = CrossEntropyLoss()  # BCELoss()
    conf.pearson_loss = partial(pearson_corr_loss, threshold=conf.sig_thresh)  # partial(pearson_corr_loss_multilabel, threshold=conf.sig_thresh)
    conf.cka_loss = CkaLoss
    conf.ncl_loss = partial(ncl_loss)
    conf.morph_loss = MSELoss()
    conf.rank_loss = CrossEntropyLoss()
    conf.rank_pearson_loss = partial(pearson_corr_loss, threshold=conf.sig_thresh)

    # xray exp
    conf.use_clean = args.clean_label
    conf.use_int = args.parenchymal
    conf.use_ext = args.extraparenchymal
    conf.ood_limit = args.ood_limit

    # skin
    conf.ood = args.ood_label
    conf.valid_size = args.valid_size

    # create learner and go
    param_desc = '_'.join(['n='+str(conf.n_models),
        str(conf.net_mode), 'lr='+str(conf.lr), 'm='+'_'.join([str(m) for m in conf.milestones]),
        ('a='+str(conf.alpha) if conf.n_models>1 else ''),
        str(conf.batch_size), 'p='+str(conf.pearson), 'mean='+str(conf.joint_mean), 'cka='+str(conf.cka)] +

        ([] if not conf.pre_train else
         ['pre', 'pre_layers='+'_'.join([str(m) for m in conf.pre_layers]),
          'pre_steps='+'_'.join([str(m) for m in conf.pre_steps])]) +

        ([] if (args.clean_label * conf.use_int * conf.use_ext) == 1 else
           ['clean-int-ext=' + str(conf.use_clean) + '-' + str(conf.use_int) + '-' + str(conf.use_ext),
            'limited=' + str(conf.ood_limit)]) +

        ([] if not args.rank else ['rank_alpha=' + str(conf.rank_alpha),
                                    'rank_pearson=' + str(conf.rank_pearson_alpha)]) +

        ([] if not args.morph else ['morph_alpha=' + str(conf.morph_alpha),
                                    'morph_func=' + conf.morph_target])

        )
    conf.log_path = str(conf.log_path) + '_' + param_desc
    conf.save_path = str(conf.save_path) + '_' + param_desc

    learner = Learner(conf)
    if conf.pre_train:
        learner.pretrain(conf)
        learner.test_eval(conf)
    else:
        learner.train(conf, conf.epochs)
        learner.test_eval(conf)

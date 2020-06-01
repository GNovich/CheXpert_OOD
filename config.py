from easydict import EasyDict as edict
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
from PIL import Image
import datetime
import time
import torch
import os

def get_config(n_models=1, logext=''):
    conf = edict()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M')

    conf.data_path = Path('data')
    conf.work_path = Path('work_space')
    conf.model_path = conf.work_path / 'models'
    if not os.path.exists(conf.work_path / 'log' / logext):
        os.mkdir(conf.work_path / 'log' / logext)
    if not os.path.exists(conf.work_path / 'save' / logext):
        os.mkdir(conf.work_path / 'save' / logext)
    conf.log_path = conf.work_path / 'log' / logext / st
    conf.save_path = conf.work_path / 'save' / logext / st

    conf.net_mode = 'resnet50'  # or 'ir

    conf.im_transform = trans.Compose([
            Image.fromarray,
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip(),
            trans.RandomRotation(30),  # subtle
            trans.Resize(225),
            trans.RandomCrop((225, 225)),
            trans.ToTensor()
    ])
    conf.data_mode = 'crop_data'
    conf.train_folder = conf.data_path / conf.data_mode / 'train'
    conf.test_folder = conf.data_path / conf.data_mode / 'test'
    conf.valid_ratio = .2
    conf.batch_size = 100  # irse net depth 50

    # --------------------Training Config ------------------------

    conf.lr = 1e-3
    conf.momentum = 0.9
    conf.pin_memory = True
    conf.num_workers = 1
    conf.ce_loss = CrossEntropyLoss()

    # additional #
    conf.cancer_only = 0
    conf.type_only = 0
    conf.no_bkg = 0
    conf.half = 0
    conf.ngpu = 1
    conf.pre_layers = []
    conf.pre_steps = []
    conf.pre_train = []
    conf.local_rank = 0
    conf.n_patch = 2
    conf.bkg_prob = .5
    conf.epoch_per_save = 100
    conf.data_mode = 'crop_data'
    conf.cpu_mode = 0
    conf.n_models = n_models
    conf.device = torch.device("cuda" if (torch.cuda.is_available() and not conf.cpu_mode) else "cpu")
    conf.with_roi = False
    return conf

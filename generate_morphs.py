from tqdm import tqdm
import os
import cv2
import time
import numpy as np
import uuid
import torch
from torch.autograd import Variable
from JobDistributor import JobDistributor
from torch.utils.data import DataLoader
from models import PreBuildConverter
import logging
import sys
import argparse
from Datasets import get_nih, get_chexpert
logging.basicConfig(level=logging.CRITICAL)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def l2_norm(input, axis=2):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def calc_emb_torch(image_tensor_batch, model, requires_grad, skip_model=False, device=0):
    #image_tensor_batch = Variable(torch.FloatTensor(image_tensor_batch), requires_grad=requires_grad).cuda(device)
    image_tensor_batch.requires_grad = True
    if skip_model:
        return None, image_tensor_batch
    model.cuda(device)
    emb = model(image_tensor_batch)[0]

    emb = l2_norm(emb, -1) # dont see why to normalize the embedding really
    return emb, image_tensor_batch


def calc_loss(embA, embB, p=2):
    return torch.pow(torch.pow(embA - embB, p).sum(dim=-1), 1. / p).mean(0)


def clip(image_tensor, start, end):
    image_tensor = torch.clamp(image_tensor, start, end)
    return image_tensor


def morph(src_img, target_imgs, model_list, iterations, lr, device=0):
    """ Updates the image to maximize outputs for n iterations """
    video_images = []

    embB_list = []
    for model in model_list:
        avg_embB = None
        for cropB in target_imgs:
            embB, _ = calc_emb_torch(cropB, model, requires_grad=False, skip_model=False, device=device)
            if avg_embB is None:
                avg_embB = embB
            else:
                avg_embB += embB
        avg_embB = avg_embB / len(target_imgs)
        embB_list.append(avg_embB)

    _, image_tensorA = calc_emb_torch(src_img, model, requires_grad=True, skip_model=True, device=device)

    for i in range(iterations):
        loss = 0
        for model_ind, model in enumerate(model_list):
            model.zero_grad()
            embA = model(image_tensorA.to(device))[0]
            embA = l2_norm(embA, -1)
            curr_loss = calc_loss(embA, embB_list[model_ind]).to(device)
            loss = loss + curr_loss
        loss = loss / len(model_list)

        # if 0:#i % 20 == 0:
        #    loss_val = loss.cpu().data.numpy()
        #    text = 'it=%d: loss=%5.3f, lr=%f' % (i, loss_val, lr)
        #    print(text)

        loss.backward(retain_graph=True)
        avg_grad = np.abs(image_tensorA.grad.data.cpu().numpy()).mean()

        norm_lr = lr / (avg_grad + 0.00001)
        lr = lr * 0.99
        image_tensorA.data -= norm_lr * image_tensorA.grad.data
        image_tensorA.data = clip(image_tensorA.data, -2.0, 2.0)
        image_tensorA.grad.data.zero_()

    unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensorA = unorm(image_tensorA.clone())
    image_tensorA.data = clip(image_tensorA.data, 0, 1)
    return image_tensorA.cpu().data.numpy()[0]


class ImageProcessor:
    def __init__(self, model_path_list=None, n_classes=0, force=0, dst_trans=None, lr=0.01, iterations=300):
        # model skeleton
        self.devices = 3 #[0,1,2,3]
        build_model = PreBuildConverter(in_channels=3, out_classes=n_classes,
                                        add_func=True, softmax=False, pretrained=False)

        # load weights
        def load_fix(target_path):
            a = torch.load(target_path, map_location=lambda storage, loc: storage.cuda(self.devices))
            fixed_a = {k.split('module.')[-1]: a[k] for k in a}
            torch.save(fixed_a, target_path)

        self.models = []
        for target_path in model_path_list:
            model = build_model.get_by_str('densenet121').cuda(self.devices)
            load_fix(target_path)
            model.load_state_dict(torch.load(target_path))
            model.train(mode=False)
            self.models.append(model)

        # process params
        self.force_regeneration = force
        self.dst_trans = dst_trans
        self.lr = lr
        self.it_num = iterations

    def process(self, sample):
        skipped = 0
        missing = 0
        failed = 0

        inputs, label_str, path = sample
        dest_img_path = self.dst_trans(path[0])
        os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)

        if not self.force_regeneration and os.path.exists(dest_img_path):
            skipped += 1
        else:
            inputs = Variable(torch.FloatTensor(inputs)).cuda(self.devices)
            dst_images = [inputs]
            src_img = cv2.blur(inputs.cpu().data.numpy()[0], (17, 17))
            im_array = np.array([src_img])
            src_img = Variable(torch.FloatTensor(im_array)).cuda(self.devices)

            morph_from0_toA = morph(src_img, dst_images, self.models, iterations=self.it_num, lr=self.lr, device=self.devices)

            dst_im = (morph_from0_toA.transpose(1, 2, 0) * 255).astype(np.uint8)
            os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
            cv2.imwrite(dest_img_path, dst_im)

        return skipped, missing, failed


def generate_morphs(proc_num, loader, params):
    generation_gpu_ids = [3]#[0, 1, 2, 3]
    distributor = JobDistributor(proc_num, generation_gpu_ids, ImageProcessor, params=params, queue_size=100000)

    for batch_i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(0), labels.to(0)
        time.sleep(0.001)
        inputs = inputs.cpu().data.numpy()[0].transpose(1, 2, 0)
        label_str = str(int(labels.cpu()[0]))
        if len(label_str) == 1:
            label_str = '0' + label_str
        distributor.push((inputs, label_str))
    distributor.push_poison_pill()

    processed = 0
    missing = 0
    failed = 0
    skipped = 0
    for metadata, res in tqdm(distributor):
        processed += 1
        skipped_i, missing_i, failed_i = res
        skipped += skipped_i
        missing += missing_i
        failed += failed_i
        if processed % 100 == 0:
            print('Processed=%d, missing=%d, failed=%d, skipped=%d' % (processed, missing, failed, skipped))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for CBIS-DDSM')
    # morph params
    parser.add_argument("-model", "--model_paths", help="path to model used for generation", type=str, nargs='+')
    parser.add_argument("-res_dir", "--res_dir", help="target output dir name (will be created)", type=str)
    parser.add_argument("-dat_mode", "--dat_mode", help="choose dataset", default='chexpert', type=str)
    parser.add_argument("-it", "--iterations", help="training it per image", default=300, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=0.01, type=float)
    parser.add_argument("-proc", "--processes_num", help="num of processes", default=36, type=int)
    parser.add_argument("-f", "--force", help="force sucees", default=0, type=int)

    # xray exp
    parser.add_argument("-use_clean", "--clean_label", help="use 'No Finding; class", default=1, type=int)
    parser.add_argument("-use_int", "--parenchymal", help="use parenchymal classes", default=1, type=int)
    parser.add_argument("-use_ext", "--extraparenchymal", help="use extraparenchymal classes", default=1, type=int)
    parser.add_argument("-ood_limit", "--ood_limit", help="exlude positive samples of ood classes", default=0, type=int)

    args = parser.parse_args()

    # get source data
    if args.dat_mode == 'nih':
        ds_morph = get_nih(morph=True)
    elif args.dat_mode.lower() == 'chexpert':
        chexpert_args = {'No_finding': args.clean_label,
                         'parenchymal': args.parenchymal,
                         'extraparenchymal': args.extraparenchymal,
                         'limit_out_labels': args.ood_limit,
                         'with_path': True,
                         'morph_gen': True
                         }
        ds_morph = get_chexpert(**chexpert_args)
    else:
        raise ValueError('no such dataset')
    args.n_classes = len(ds_morph.label_names)

    # and loader
    dloader_args = {
        'batch_size': 1,
        'pin_memory': False,
        'num_workers': 0,
        'drop_last': False,
    }
    loader = DataLoader(ds_morph, **dloader_args)


    def path_to_morph(path):
        # tranform an image path to target path
        splitted_path = os.path.relpath(path, 'data').split('/')
        dst_list = [os.path.relpath('data'), splitted_path[0] + '_morph'] + splitted_path[1:]
        return os.path.join(*dst_list)

    params = {'model_path_list': args.model_paths,
              'n_classes': args.n_classes,
              'force': args.force,
              'dst_trans': path_to_morph,
              'lr': args.lr,
              'iterations': args.iterations
    }

    proc = ImageProcessor(**params)

    # test for a single batch
    missing = 0
    failed = 0
    skipped = 0

    #generate_morphs(args.processes_num, loader, params=params)
    for processed, sample in enumerate(tqdm(loader, total=len(loader))):
        res = proc.process(sample)
        skipped_i, missing_i, failed_i = res
        skipped += skipped_i
        missing += missing_i
        failed += failed_i
        if processed % 100 == 0:
            print('Processed=%d, missing=%d, failed=%d, skipped=%d' % (processed, missing, failed, skipped))

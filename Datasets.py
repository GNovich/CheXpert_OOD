import torch
from PIL import Image
from torchvision import transforms as trans
from torch.utils.data import Dataset
import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

ImageNet_norm = trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ImageNet_trans = {'train':
    trans.Compose([
        trans.RandomResizedCrop(224),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        ImageNet_norm,
    ]),
    'test':
    trans.Compose([
        trans.Resize(256),
        trans.TenCrop(224),
        trans.Lambda(lambda crops: torch.stack([trans.ToTensor()(crop) for crop in crops])),
        trans.Lambda(lambda crops: torch.stack([ImageNet_norm(crop) for crop in crops]))
    ])
}


class NIH(Dataset):
    """
        https://arxiv.org/pdf/1711.05225.pdf
    """
    def __init__(self, table, transform=None):
        self.og_idices = table.index.values
        self.image_names = table['Image Index'].values
        self.paths = table['path'].values
        self.label_encoding = table['Finding Labels'].str.get_dummies(sep='|')
        self.label_names = self.label_encoding.columns.values
        self.labels = torch.FloatTensor(self.label_encoding.values)
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label.float()

    def __len__(self):
        return len(self.paths)


def get_nih(seed=2020):
    all_xray_df = pd.read_csv('data/NIH/Data_Entry_2017.csv')
    all_image_paths = {os.path.basename(x): x for x in
                       glob(os.path.join('data/NIH', 'images*', '*', '*.png'))}
    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

    train_df, test_df = train_test_split(all_xray_df,
                                         test_size=0.30,
                                         random_state=seed)
    # stratify = all_xray_df[['Cardiomegaly', 'Patient Gender']])
    return NIH(train_df, transform=ImageNet_trans['train']), NIH(test_df, transform=ImageNet_trans['test'])



class CheXpert(Dataset):
    def __init__(self, table, transform=None, policy=1):
        self.table = table
        self.label_names = table.columns[2:].values
        self.paths = table['Path'].values
        self.mapping = dict({1: 1, 0: 0, -1: policy})
        self.labels = torch.FloatTensor(table[self.label_names].fillna(0).applymap(lambda x: self.mapping[x]).values)
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label.float()

    def __len__(self):
        return len(self.paths)


def get_chexpert(seed=2020, policy=1,
                 No_finding=True, parenchymal=True, extraparenchymal=True, limit_out_labels=True):
    Parenchymal = [  # i.e. findings in the lungs themselves
        'Lung Lesion',
        'Lung Opacity',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
    ]

    Extraparenchymal = [  # i.e. findings outside the lungs
        'Support Devices',
        'Pleural Effusion',
        'Pleural Other',
        'Pneumothorax',
        'Cardiomegaly',
        'Enlarged Cardiomediastinum',
        'Fracture',
    ]

    labels = ['No Finding'] if No_finding else []
    labels = labels + Parenchymal if parenchymal else labels
    labels = labels + Extraparenchymal if extraparenchymal else labels

    out_labels = []  # 'No Finding' can never be OOD, as it is alway in-dist phen
    out_labels = out_labels + Parenchymal if not parenchymal else out_labels
    out_labels = out_labels + Extraparenchymal if not extraparenchymal else out_labels

    cheXpert_train = pd.read_csv('data/CheXpert-v1.0-small/train.csv')
    cheXpert_test = pd.read_csv('data/CheXpert-v1.0-small/valid.csv')
    cheXpert = pd.concat([cheXpert_train, cheXpert_test]).reset_index(drop=True)
    cheXpert['Path'] = cheXpert['Path'].apply(lambda x: 'data/' + x)
    cheXpert['Patient ID'] = cheXpert['Path'].str.extract(r'patient([0-9]+)\/').astype(int).values
    column_set = ['Path', 'Patient ID'] + labels

    # drop lateral views (eq to Frontal/Lateral == Frontal)
    cheXpert = cheXpert[~cheXpert['AP/PA'].isna()]

    # label policy - uncertine labels should be counted as:
    mapping = dict({1: 1, 0: 0, -1: policy})
    cheXpert[labels + out_labels] = cheXpert[labels + out_labels].fillna(0).applymap(lambda x: mapping[x]).values

    # limit dataset to places with no positive from the out labels
    if limit_out_labels:
        ood_exists = cheXpert[out_labels].sum(axis=1) > 0
        cheXpert_out = cheXpert[ood_exists]
        in_exists = cheXpert_out[labels].sum(axis=1) > 0
        cheXpert_out_in = cheXpert_out[in_exists]
        cheXpert_out_out = cheXpert_out[~in_exists]
        cheXpert = cheXpert[~ood_exists]
    else:
        ood_exists = (cheXpert[out_labels].sum(axis=1) > 0) & (cheXpert[labels].sum(axis=1) < 1)
        cheXpert_out = cheXpert[ood_exists]
        cheXpert = cheXpert[~ood_exists]
        cheXpert_out_in = pd.DataFrame([], columns=cheXpert_out.columns)
        cheXpert_out_out = cheXpert_out

    # split test train by identities
    splitter = GroupShuffleSplit(1, test_size=.3, random_state=seed)
    id_locs = list(splitter.split(range(len(cheXpert)), groups=cheXpert['Patient ID']))[0]
    train_df, test_df = cheXpert.iloc[id_locs[0]][column_set], cheXpert.iloc[id_locs[1]][column_set]

    return (CheXpert(train_df, transform=ImageNet_trans['train'], policy=policy),
            CheXpert(test_df, transform=ImageNet_trans['test'], policy=policy),
            cheXpert_out_in,
            cheXpert_out_out)
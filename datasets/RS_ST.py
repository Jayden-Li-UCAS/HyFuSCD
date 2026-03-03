import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import datasets.transform as transform
from torchvision.transforms import functional as F


DATASET_CONFIGS = {
    "SECOND": {
        "ST_COLORMAP": [[255,255,255], [0,128,0], [128,128,128], [0,255,0], [0,0,255], [128,0,0], [255,0,0]],
        "ST_CLASSES": ['unchanged', 'low vegetation', 'ground', 'tree', 'water', 'building', 'sports field']
    },
    "Landsat": {
        "ST_COLORMAP": [[255,255,255], [0,155,0], [255,165,0], [230,30,100], [0,170,240]],
        "ST_CLASSES": ['unchanged', 'farmland', 'desert', 'building', 'water']
    },
    "Map_SCD": {
        "ST_COLORMAP": [[0, 0, 0],  [0,0,128], [121, 181, 70], [148, 148, 148], [70, 184, 222], [189, 140, 28], [27, 187, 167], [70, 70, 181]],
        "ST_CLASSES": ['unchanged',  'bareland', 'vegetation', 'developed spaces', 'road', 'water', 'cropland', 'building']
    },
    "SCSCD7": {
        "ST_COLORMAP": [[255, 255, 255], [49, 125, 237], [255,0,0], [0, 0, 255], [0, 255, 255], [0, 255, 0], [0, 128, 0], [128, 128, 128]],
        "ST_CLASSES": ['unchanged', 'bareland', 'water', 'building', 'structure', 'farmland', 'vegetation', 'road']
    }
}

def get_dataset_config(dataname):
    if dataname not in DATASET_CONFIGS:
        raise ValueError(f"数据集{dataname}未配置！请在dataset_config.py的DATASET_CONFIGS中添加")
    config = DATASET_CONFIGS[dataname].copy()
    config["num_classes"] = len(config["ST_CLASSES"])
    return config

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time == 'A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im


class Data(data.Dataset):
    def __init__(self, datapath, mode, augmentation=False):
        self.datapath = datapath
        self.mode = mode
        self.augmentation = augmentation

        self.A = os.path.join(datapath, self.mode, "im1")
        self.B = os.path.join(datapath, self.mode,"im2")
        self.labels_A = os.path.join(datapath, self.mode,"label1")
        self.labels_B = os.path.join(datapath, self.mode,"label2")

        self.list_img = self.get_mask_name(datapath)

    def get_mask_name(self, datapath):
        images_list_file = os.path.join(datapath, 'list', self.mode + ".txt")
        with open(images_list_file, "r") as f:
            return f.readlines()


    def __getitem__(self, idx):
        imgname = self.list_img[idx].strip('\n')

        img_A = io.imread(os.path.join(self.A, imgname))
        img_B = io.imread(os.path.join(self.B, imgname))
        label_A = io.imread(os.path.join(self.labels_A, imgname))
        label_B = io.imread(os.path.join(self.labels_B, imgname))

        if self.augmentation:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_MCD(img_A, img_B, label_A, label_B)

        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B), \
               torch.from_numpy(label_A), torch.from_numpy(label_B),imgname

    def __len__(self):
        return len(self.list_img)




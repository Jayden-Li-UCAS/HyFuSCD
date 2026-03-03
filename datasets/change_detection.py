from datasets.augmentation import augmentation_compose
import numpy as np
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES_SECOND = ['unchanged', 'water', 'ground', 'Low Vegetation', 'tree', 'building', 'Sports Filed']
CLASSES_Landsat = ['unchanged', 'farmland', 'desert', 'building', 'water']
CLASSES_SCSCD7 = ['unchanged', 'bareland', 'water', 'building',
                  'structure', 'farmland', 'vegetation', 'road']
CLASSES_OpenMapCD = ['unchanged', 'bareland', 'vegetation', 'developed spaces', 'road', 'water', 'cropland', 'building']

DATASET_CLASSES_MAP = {
    "SECOND": CLASSES_SECOND,
    "Landsat": CLASSES_Landsat,
    "SCSCD7": CLASSES_SCSCD7,
    "OpenMapCD": CLASSES_OpenMapCD
}

def get_num_classes(data_name):
    if data_name not in DATASET_CLASSES_MAP:
        raise ValueError(f"Invalid dataset name: {data_name}, available options: {list(DATASET_CLASSES_MAP.keys())}")
    return len(DATASET_CLASSES_MAP[data_name])


class ChangeDetection(Dataset):
    def get_num_classes(data_name):
        if data_name not in DATASET_CLASSES_MAP:
            raise ValueError(f"Invalid dataset name: {data_name}, available options: {list(DATASET_CLASSES_MAP.keys())}")
        return len(DATASET_CLASSES_MAP[data_name])

    def __init__(self, root, mode):
        super(ChangeDetection, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'im2', id)))

        mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id)))

        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0] = 1

        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                     'gt_mask': mask_bin})
            img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                                                 sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)
from hydra.utils import instantiate
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import glob
from torchvision import transforms



class ExFractal_dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        data_ls = []
        label_ls=[]
        for dir_path in glob.glob(f"{img_dir}/*"):
            view_ls = []
            base_path, dir_name =dir_path.rsplit('/', 1)
            for data_path in sorted(glob.glob(f"{self.img_dir}/{dir_name}/*.png")):
                view_ls.append(data_path)
            #add list of multi view point imagelist to datalist
            data_ls.append(view_ls)
            #add label to list
            label_ls.append(int(dir_name))
            
        self.data = data_ls
        self.label = label_ls
        
        self.len = len(label_ls)
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        #select data
        data = self.data[index]
        #select view point
        dice = np.random.randint(0,len(data),1)
        data_path = data[dice[0]]

        image = Image.open(data_path)
        image = image.convert('RGB')
        #select label
        label = self.label[index]
        
        #transform data
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label).long()
            
        return image, label


def create_dataloader(data_cfg, is_training=True):
    transform = instantiate(data_cfg.transform, is_training=is_training)
    print(f'Data augmentation is as follows \n{transform}\n')

    if is_training:
        print("ok",data_cfg.baseinfo.name)
        if "3d-ofdb" in data_cfg.baseinfo.name:
            print("dataset's name include 3d-ofdb")
            print(data_cfg.trainset.root)
            dataset = ExFractal_dataset(data_cfg.trainset.root, transform=transform)
            print(len(dataset))
        else:
            dataset = instantiate(data_cfg.trainset, transform=transform)
        
        print(f'{len(dataset)} images and {data_cfg.baseinfo.num_classes} classes were found from {data_cfg.trainset.root}')
    else:
        dataset = instantiate(data_cfg.valset, transform=transform)
        print(f'{len(dataset)} images and {len(dataset.classes)} classes were found from {data_cfg.valset.root}')

    sampler = instantiate(data_cfg.sampler, dataset=dataset, shuffle=is_training)
    dataloader = instantiate(data_cfg.loader, dataset=dataset, sampler=sampler, drop_last=is_training)
    return dataloader


# Adapted from https://github.com/annegnx/PnP-Flow

import torch
import torchvision.transforms as v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import warnings
import logging
import socket
from pathlib import Path

class DataLoaders:
    def __init__(self, dataset_name, batch_size_train, batch_size_test, dim_image=128, train=False):
        self.dataset_name = dataset_name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.base_path = Path(__file__).resolve().parent.parent.parent
        self.dim_image = dim_image
        self.train = train

    def load_data(self):
        if self.dataset_name == 'celeba':
            transform = v2.Compose([
                v2.CenterCrop(178),
                v2.Resize((128, 128)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            # Paths
            img_dir = os.path.join(self.base_path, 'natural', 'data', 'celeba', 'img_align_celeba', 'img_align_celeba')
            partition_csv = os.path.join(self.base_path, 'natural', 'data', 'celeba', 'list_eval_partition.csv')

            # Datasets
            train_dataset = CelebADataset(img_dir, partition_csv, partition=0, transform=transform)
            val_dataset = CelebADataset(img_dir, partition_csv, partition=1, transform=transform)
            test_dataset = CelebADataset(img_dir, partition_csv, partition=2, transform=transform)

            shuffle_train = len(train_dataset) > 0

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=shuffle_train,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'afhq_cat':
            # transform should include a linear transform 2x - 1
            transform = v2.Compose([
                v2.Resize((256, 256)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            img_dir_train = os.path.join(self.base_path, 'natural', 'data', 'afhq_cat', 'train', 'cat')
            img_dir_val = os.path.join(self.base_path, 'natural', 'data', 'afhq_cat', 'val', 'cat')
            img_dir_test = os.path.join(self.base_path, 'natural', 'data', 'afhq_cat', 'test', 'cat')

            train_dataset = AFHQDataset(img_dir_train, batchsize=self.batch_size_test, transform=transform)
            val_dataset = AFHQDataset(img_dir_val, batchsize=self.batch_size_test, transform=transform)
            test_dataset = AFHQDataset(img_dir_test, batchsize=self.batch_size_test, transform=transform)

            shuffle_train = len(train_dataset) > 0

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=shuffle_train,
                collate_fn=custom_collate, drop_last=True)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'coco':
            transform = v2.Compose([
                v2.Resize((self.dim_image, self.dim_image), interpolation=v2.InterpolationMode.BILINEAR),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            img_dir_train = os.path.join(self.base_path, 'natural', 'data', 'coco', 'train')
            img_dir_val = os.path.join(self.base_path, 'natural', 'data', 'coco', 'val')
            img_dir_test = os.path.join(self.base_path, 'natural', 'data', 'coco', 'test')

            train_dataset = COCODataset(img_dir_train, batchsize=self.batch_size_test, transform=transform)
            val_dataset = COCODataset(img_dir_val, batchsize=self.batch_size_test, transform=transform)
            test_dataset = COCODataset(img_dir_test, batchsize=self.batch_size_test, transform=transform)

            shuffle_train = len(train_dataset) > 0

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=shuffle_train,
                collate_fn=custom_collate, drop_last=True)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
        else:
            raise ValueError("The dataset your entered does not exist")

        data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        return data_loaders


class CelebADataset(Dataset):
    def __init__(self, img_dir, partition_csv, partition, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.partition = partition

        # Load the partition file correctly
        partition_df = pd.read_csv(
            partition_csv, header=0, names=[
                'image', 'partition'], skiprows=1)
        self.img_names = partition_df[partition_df['partition']
                                      == partition]['image'].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class AFHQDataset(Dataset):
    """AFHQ Cat dataset."""

    def __init__(self, img_dir, batchsize, category='cat', transform=None):
        self.files = os.listdir(img_dir)
        self.num_imgs = len(self.files)
        self.batchsize = batchsize
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class COCODataset(Dataset):
    """COCO dataset."""
    def __init__(self, img_dir, batchsize, transform=None):
        self.files = os.listdir(img_dir)
        self.num_imgs = len(self.files)
        self.batchsize = batchsize
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


def custom_collate(batch):
    # Filter out None values

    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data._utils.collate.default_collate(batch)


logging.basicConfig(level=logging.INFO)

from typing import Optional, List
import copy
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from pytorch_lightning import seed_everything

import wilds
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

import open_clip
import torch.nn as nn
import pyrootutils
import random

import os
import pandas as pd

from src.datasets.labels import _D2LABEL
from src.datasets.prompt_template import _TEMPLATE


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datasets.dataloader import get_adaptive_train_loader
from src.datasets.wilds_datasets_cfgs import _WILDS_DATASETS
from src.datasets.multi_source_domain_net import MultiSourceDomainNetDataset
import src.utils.logging as logging

logger = logging.get_logger("smart_canada_goose")

from wilds.datasets.poverty_dataset import _MEANS_2009_17, _STD_DEVS_2009_17

poverty_rgb_means = torch.from_numpy(np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1)))
poverty_rgb_stds = torch.from_numpy(np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1)))

def poverty_rgb_color_transform(ms_img, image_size, transform=None):
    poverty_rgb_means = torch.from_numpy(
        np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1)))
    poverty_rgb_stds = torch.from_numpy(
        np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1)))

    poverty_rgb_means = poverty_rgb_means.to(device=ms_img.device)
    poverty_rgb_stds = poverty_rgb_stds.to(device=ms_img.device)
    def unnormalize_rgb_in_poverty_ms_img(ms_img):
        result = ms_img.detach().clone()
        result = (result * poverty_rgb_stds) + poverty_rgb_means
        return result

    color_transform = transforms.Compose([
        transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),

    ])
    # The first three channels of the Poverty MS images are BGR
    # So we shuffle them to the standard RGB to do the ColorJitter
    # Before shuffling them back
    ms_img = color_transform(ms_img[:, [2,1,0]]) # bgr to rgb to bgr

    return ms_img
    # return ms_img

def initialize_image_base_transform(dataset):
    transform_steps = []
    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))
    transform_steps.append(transforms.Resize((448, 448)))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transform_steps)
    return transform


def build_open_clip_transform(image_size=448, is_train=False):
    return open_clip.image_transform(
        image_size,
        is_train=is_train,
    )

def _convert_to_rgb(image):
    return image.convert('RGB')

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def WILD_transform(image_size=448):

    return transforms.Compose([transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.RandomHorizontalFlip(),
                               transforms.CenterCrop(image_size),
                               RandomApply(transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
                               _convert_to_rgb,
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                    std=(0.26862954, 0.26130258, 0.27577711)),
                               ])


def fmow_transform(image_size=448):


    return transforms.Compose([

        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               transforms.CenterCrop(image_size),
                               _convert_to_rgb,
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711)),
                               ])

def fmow_test(image_size=448):


    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(image_size),
                               _convert_to_rgb,
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711)),
                               ])

def domain_transform(image_size=448):

    return transforms.Compose(
            [
                # transforms.Resize(256),384
                transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711)),
            ])

def domain_test(image_size=448):


    return transforms.Compose([
        transforms.Resize((image_size,image_size)),
                               _convert_to_rgb,
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711)),
                               ])



def get_subset_with_domain_id(dataset, grouper, domain=None):
    if type(dataset) != wilds.datasets.wilds_dataset.WILDSSubset:
        raise NotImplementedError
    subset = copy.deepcopy(dataset)

    if domain is not None:
        # TODO: Hard code to fix [0]
        domain_name = grouper.groupby_fields[0]

        domain_idx = dataset.metadata_fields.index(domain_name)

        idx = np.argwhere(
            np.isin(subset.dataset.metadata_array[:, domain_idx][subset.indices], domain)
        ).ravel()
        subset.indices = subset.indices[idx]
    else:
        raise NotImplementedError

    return subset


def get_test_loaders(dataset, grouper, batch_size=16, num_workers=0):
    all_domain_ids = list(
        set(
            grouper.metadata_to_group(
                dataset.dataset.metadata_array[dataset.indices]
            ).tolist()
        )
    )
    test_domain_loaders = []

    for domain in all_domain_ids:
        domain_data = get_subset_with_domain_id(dataset, grouper, domain=domain)
        domain_loader = get_eval_loader(
            "standard", domain_data, batch_size=batch_size, num_workers=num_workers
        )
        test_domain_loaders.append(domain_loader)

    return test_domain_loaders




_TYPE_DATALOADERS = ["erm", "adaptive"]


class WILDS2LabeledDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str = "/data01/tao/wilds/data",
        dataset_name: str = "iwildcam",
        input_resolution: int = 448,
        domain_type: str = "erm",
        batch_size: int = 16,
        num_workers: int = 0,
    ):
        super().__init__()
        """
                domain_type (str): make a choice between "erm" and "adapt"
            """
        if dataset_name not in _WILDS_DATASETS:
            raise ValueError(
                f"Please choose a dataset from {list(_WILDS_DATASETS.keys())}"
            )

        if domain_type not in _TYPE_DATALOADERS:
            raise ValueError(
                f"Please choose a dataloader type from {_TYPE_DATALOADERS}"
            )

        self.data_path = data_path
        self.dataset = _WILDS_DATASETS[dataset_name]

        self.input_resolution = input_resolution
        self.domain_type = domain_type
        self.batch_size = batch_size
        self.num_workers = num_workers


    def poverty_rgb_color_transform(self, ms_img, is_train=False):
        poverty_rgb_means = torch.from_numpy(
            np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1)))
        poverty_rgb_stds = torch.from_numpy(
            np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1)))

        poverty_rgb_means = poverty_rgb_means.to(device=ms_img.device)
        poverty_rgb_stds = poverty_rgb_stds.to(device=ms_img.device)

        def unnormalize_rgb_in_poverty_ms_img(ms_img):
            result = ms_img.detach().clone()
            result = (result * poverty_rgb_stds) + poverty_rgb_means
            return result
        if is_train:
            color_transform = transforms.Compose([
                transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
                transforms.RandomResizedCrop(self.input_resolution, scale=(0.5, 1.0),
                                            interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            color_transform = transforms.Compose([
            transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
            transforms.Resize(self.input_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        # The first three channels of the Poverty MS images are BGR
        # So we shuffle them to the standard RGB to do the ColorJitter
        # Before shuffling them back
        # transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1)
        ms_img = color_transform(ms_img[:, [2, 1, 0]])  # bgr to rgb to bgr

        return ms_img

    def setup(self, stage: Optional[str] = None):
        

        if "domainnet" in self.dataset.name:
            label_names = _D2LABEL['domainnet']
            p_template = _TEMPLATE['imagenet']

        elif "iwildcam" in self.dataset.name:
            labels_csv = os.path.join('src/datasets', 'iwildcam_categories.csv')
            df = pd.read_csv(labels_csv)
            df = df[df['y'] < 99999]

            label_names = [s.lower() for s in list(df['english'])]
            p_template = _TEMPLATE['iwildcam'] + ["{} in the wild."]

        elif "fmow" in self.dataset.name:
            label_names = _D2LABEL['fmow']
            p_template = _TEMPLATE['fmow']

        elif 'camelyon' in self.dataset.name:
            label_names = _D2LABEL['camelyon']
            p_template = _TEMPLATE['camelyon']

        elif 'poverty' in self.dataset.name:
            label_names = _D2LABEL['poverty']
            p_template = _TEMPLATE['poverty']

        elif 'rxrx1' in self.dataset.name:
            label_names = _D2LABEL['rxrx1']
            p_template = _TEMPLATE['rxrx1']


        self.label_names = label_names
        self.p_template = p_template



        if "domainnet" in self.dataset.name:
            datasets = MultiSourceDomainNetDataset(
                download=False, target_domain=self.dataset.target_domain, root_dir=self.data_path
            )
        else:
            datasets = get_dataset(
                dataset=self.dataset.name, download=False, root_dir=self.data_path
            )

        if self.dataset.name == 'poverty':
            train_transform = None
            val_transform = None
            self.uniform_over_group = True
            self.uniform_sampler = True
            
            self.post_transform = self.poverty_rgb_color_transform


        elif "domainnet" or 'rxrx1' in self.dataset.name:
            self.uniform_over_group = False
            self.uniform_sampler = False
            train_transform = domain_transform(
                image_size=self.input_resolution)

            val_transform = build_open_clip_transform(
                image_size=self.input_resolution, is_train=False
            )
        elif "fmow" in self.dataset.name:
            # default False, False
            self.uniform_over_group = False
            self.uniform_sampler = False
            train_transform = fmow_transform(
                image_size=self.input_resolution
            )

            val_transform = fmow_test(
                image_size=self.input_resolution
            )
        elif "camelyon" in self.dataset.name:

            self.uniform_over_group = True
            self.uniform_sampler = True
            train_transform = WILD_transform(
                image_size=self.input_resolution
            )
            val_transform = build_open_clip_transform(image_size=self.input_resolution, is_train=False)

        else:

            self.uniform_over_group = True
            self.uniform_sampler = True
            train_transform = WILD_transform(
                image_size=self.input_resolution
            )
            val_transform = build_open_clip_transform(image_size=self.input_resolution, is_train=False)

        self.grouper = CombinatorialGrouper(datasets, [self.dataset.domain_name])
        if self.dataset.name == 'fmow':
            print(':::: choosing dataset:', self.dataset.name)
            self.test_grouper = CombinatorialGrouper(datasets, [self.dataset.test_domain_name])

        # Train-set
        self.train_dataset = datasets.get_subset(
            self.dataset.train_split, transform=train_transform
        )

        print(self.train_dataset._metadata_fields)
        self.train_domain_ids = list(
            set(
                self.grouper.metadata_to_group(
                    self.train_dataset.dataset.metadata_array[
                        self.train_dataset.indices
                    ]
                ).tolist()
            )
        )
        logger.info(
            f"{self.dataset.name}'s train-set has {len(self.train_dataset)} data examples, {self.train_dataset.n_classes} classes and {len(self.train_domain_ids)} domains"
        )

        if self.dataset.id_test_split:
            self.id_test_dataset = datasets.get_subset(
                self.dataset.id_test_split, transform=val_transform
            )
            self.id_test_domain_ids = list(
                set(
                    self.grouper.metadata_to_group(
                        self.id_test_dataset.dataset.metadata_array[
                            self.id_test_dataset.indices
                        ]
                    ).tolist()
                )
            )
            logger.info(
                f"{self.dataset.name}'s id-test-set has {len(self.id_test_dataset)} data examples, {self.id_test_dataset.n_classes} classes and {len(self.id_test_domain_ids)} domains"
            )
            print('check ID domains: ', self.id_test_domain_ids)

        self.ood_test_dataset = datasets.get_subset(
            self.dataset.ood_test_split, transform=val_transform
        )

        self.ood_test_domain_ids = list(
            set(
                self.grouper.metadata_to_group(
                    self.ood_test_dataset.dataset.metadata_array[
                        self.ood_test_dataset.indices
                    ]
                ).tolist()
            )
        )

        logger.info(
            f"{self.dataset.name}'s ood-test-set has {len(self.ood_test_dataset)} data examples, {self.ood_test_dataset.n_classes} classes and {len(self.ood_test_domain_ids)} domains"
        )

        print('check OOD domains: ', self.ood_test_domain_ids)

    def train_dataloader(self) -> DataLoader:
        if self.domain_type == "erm":
            loader = get_train_loader(
                "standard",
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        elif self.domain_type == "adaptive":
            loader = get_adaptive_train_loader(
                self.train_dataset,
                grouper=self.grouper,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                uniform_over_groups=self.uniform_over_group,
                uniform_sampler= self.uniform_sampler,
            )
        else:
            raise NotImplementedError
        return loader

    def val_dataloader(self) -> List[DataLoader]:
        """
        The doc for combining dataloaders: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.utilities.combined_loader.html


        """
        loaders = []
        # return CombinedLoader(loaders, "sequential")
        ood_loaders = get_test_loaders(
            self.ood_test_dataset,
            self.grouper,
            batch_size=self.batch_size,
            num_workers=0,
        )

        loaders += [ood_loaders]


        if self.dataset.id_test_split:
            id_loaders = get_test_loaders(
                self.id_test_dataset,
                self.grouper,
                batch_size=self.batch_size,
                num_workers=0,
            )

            loaders += id_loaders

        return CombinedLoader(loaders, "sequential")


if __name__ == "__main__":
    seed_everything(42)
    bs = 64
    datamodule = WILDS2LabeledDataModule(
        data_path="/data/dataset",
        dataset_name="domain_net_clipart",
        domain_type="erm",
        batch_size=bs,
        num_workers=8,
        input_resolution=224
        # n_negative_groups_per_batch=1,
        # n_points_per_negative_group= 16,
    )
    datamodule.setup()

    print(datamodule.label_prompt)

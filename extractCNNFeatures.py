from __future__ import print_function, division

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
import os

import scipy.io as io
import fnmatch
import wrgbd51
import collections
import argparse


# Data augmentation and normalization for training
# Just normalization for validation
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'test']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                              shuffle=True, num_workers=4)
#               for x in ['train', 'test']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
# class_names = image_datasets['train'].classes
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WashingtonDataset(Dataset):
    def __init__(self, data_path, split_path, data_type='crop', split=1, transform=None):
        self.data_path = data_path
        self.data_type = data_type
        self.split_path = split_path
        self.split = split
        self.transform = transform
        self.data = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.paths = collections.defaultdict(list)
        self._init_dataset()

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self, other):
        raise NotImplementedError

    def _init_dataset(self):
        split_data = io.loadmat(self.split_path)['splits'].astype(np.uint8)
        test_instances = split_data[:, self.split - 1]

        suffix = '*_' + self.data_type + ".png"
        for category in os.listdir(self.data_path):
            category_path = os.path.join(self.data_path, category)
            cat_ind = int(wrgbd51.class_name_to_id[category])

            for instance in os.listdir(category_path):
                instance_path = os.path.join(category_path, instance)
                # print('c: {} : {} i: {} {} token: {}'.format(cat_ind, category, i, instance, instance.split('_')[-1]))

                for file in fnmatch.filter(os.listdir(instance_path), suffix):
                    if test_instances[cat_ind - 1] == np.uint8(instance.split('_')[-1]):
                        self.data['test'].append(file)
                        self.paths['test'].append(os.path.join(instance_path, file))
                        self.labels['test'].append(cat_ind)
                        print(os.path.join(instance_path, file))
                    else:
                        self.data['train'].append(file)
                        self.paths['train'].append(os.path.join(instance_path, file))
                        self.labels['train'].append(cat_ind)


def get_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset-path", dest="dataset_path", default="/media/ali/ssdmain/Datasets/wrgbd",
                        help="Path to dataset root")
    parser.add_argument("--data-dir", dest="data_dir", default="eval-set",
                        help="Data dir")
    parser.add_argument("--split-file", dest="split_file", default="splits.mat", help="split file name, must be under"
                                                                                      " --dataset-path")
    parser.add_argument("--data-type", dest="data_type", default="crop", choices=['crop', 'depthcrop', 'maskcrop'],
                        type=str.lower, help="Data type to process")

    options = parser.parse_args()
    return options


def main():
    options = get_options()
    data_dir = os.path.join(options.dataset_path, options.data_dir)
    split_file = os.path.join(options.dataset_path, options.split_file)

    wrgbd = WashingtonDataset(data_dir, split_file, data_type=options.data_type, split=1)


if __name__ == '__main__':
    main()

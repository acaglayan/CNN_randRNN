from __future__ import print_function, division

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms, utils
import os

import scipy.io as io
import fnmatch
import wrgbd51
from PIL import Image
import argparse
import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WashingtonDataset(Dataset):
    def __init__(self, data_path, split_file, data_type, split, mode, transform=None, loader=None):
        self.data_path = data_path
        self.data_type = data_type
        self.split_file = split_file
        self.split = split
        self.mode = mode
        self.transform = transform
        self.loader = loader
        data = self._init_dataset()
        self.data = data
        self.targets = [d[1] for d in data]

    def __getitem__(self, index):
        path, target = self.data[index]
        datum = self.loader(path, self.data_type)

        if self.transform is not None:
            datum = self.transform(datum)

        return datum, target

    def __len__(self):
        return len(self.data)

    def _init_dataset(self):
        images = []
        split_data = io.loadmat(self.split_file)['splits'].astype(np.uint8)
        test_instances = split_data[:, self.split - 1]

        for category in os.listdir(self.data_path):

            category_path = os.path.join(self.data_path, category)
            cat_ind = int(wrgbd51.class_name_to_id[category])

            for instance in os.listdir(category_path):
                instance_path = os.path.join(category_path, instance)

                if self.mode == 'test':
                    if test_instances[cat_ind - 1] == np.uint8(instance.split('_')[-1]):
                        images.extend(self.add_item(instance_path, cat_ind))

                elif self.mode == 'train':
                    if test_instances[cat_ind - 1] != np.uint8(instance.split('_')[-1]):
                        images.extend(self.add_item(instance_path, cat_ind))

        return images

    def add_item(self, instance_path, cat_ind):
        images = []
        suffix = '*_' + self.data_type + '.png'
        for file in fnmatch.filter(os.listdir(instance_path), suffix):
            path = os.path.join(instance_path, file)
            item = (path, cat_ind)
            images.append(item)
        return images


def colorized_depth(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        # img = img.astype("float64")

        # source: https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
        zy, zx = np.gradient(img)
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        # zx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        # zy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        normal = np.dstack((-zx, -zy, np.ones_like(img)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

        # offset and rescale values to be in 0-255
        normal += 1
        normal /= 2
        normal *= 255

        return np.uint8(normal[:, :, ::-1])


def pil_loader(path, data_type):
    if data_type == 'depthcrop':
        img = colorized_depth(path)
        return Image.fromarray(img).convert('RGB')
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset-path", dest="dataset_path", default="/media/ali/ssdmain/Datasets/wrgbd/",
                        help="Path to the dataset root")
    parser.add_argument("--data-path", dest="data_dir", default="eval-set", help="Data dir")
    parser.add_argument("--split-file", dest="split_file", default="splits.mat", help="split file name, must be under "
                                                                                      "--dataset-path")
    parser.add_argument("--data-type", dest="data_type", default="crop", choices=['crop', 'depthcrop'], type=str.lower,
                        help="data type to process, crop for rgb, depthcrop for depth data")
    parser.add_argument("--split", dest="split", default=1, type=int, choices=range(1, 11),
                        help="current split number to process, is between 1 to 10")
    parser.add_argument("--mode", dest="mode", default="test", choices=['train', 'test'], type=str.lower,
                        help="data split mode, train or test")

    params = parser.parse_args()
    return params


def imshow(inp, title=None):
    """imshow for tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


def main():
    params = get_params()
    data_dir = os.path.join(params.dataset_path, params.data_dir)
    split_file = os.path.join(params.dataset_path, params.split_file)

    train_form = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    training_set = WashingtonDataset(data_dir, split_file, data_type='depthcrop', split=1, mode='train',
                                     loader=pil_loader, transform=train_form)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)

    test_form = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_set = WashingtonDataset(data_dir, split_file, data_type='crop', split=1, mode='test',
                                 loader=pil_loader, transform=test_form)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)


    # for i, data_samples in enumerate(train_loader):
    #    img, target = data_samples

    print(training_set.__len__())

    inputs, classes = next(iter(train_loader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[wrgbd51.class_id_to_name[str(x)] for x in np.array(classes)])


if __name__ == '__main__':
    main()

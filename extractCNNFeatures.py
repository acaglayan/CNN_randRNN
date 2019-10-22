from __future__ import print_function, division

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import svm
from torch.nn import init
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms, models, utils
import os

import scipy.io as io
import fnmatch
import wrgbd51
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import time


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


class AlexNetExtractor(nn.Module):
    def __init__(self, submodule, extracted_layer):
        super(AlexNetExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layer = extracted_layer
        if self.extracted_layer < 6:
            self.features = self._get_features()

    def forward(self, x):
        if self.extracted_layer < 6:
            x = self.features(x)
        else:
            self.submodule.classifier = self._get_classifier()
            x = self.submodule(x)
        return x

    def _get_features(self):
        index = self._find_index()
        features = nn.Sequential(
            # stop at the layer
            *list(self.submodule.features.children())[:index]
        )
        return features

    def _get_classifier(self):
        index = self._find_index()
        classifier = nn.Sequential(
            # stop at the layer
            *list(self.submodule.classifier.children())[:index]
        )
        return classifier

    def _find_index(self):
        switcher = {
            1: 3,   # from features
            2: 6,
            3: 8,
            4: 10,
            5: 13,
            6: 3,   # from classifier
            7: 6
        }
        return switcher.get(self.extracted_layer)


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
    plt.pause(0.001)  # pause a bit so that plots are updated


"""
a general function to train a model. Here, we will illustrate:

    Scheduling the learning rate
    Saving the best model
parameter scheduler is an LR scheduler object from torch.optim.lr_scheduler
"""


def train_eval_model(model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # each epoch has a training and evaluation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best eval Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, data_loaders, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(wrgbd51.class_id_to_name[np.array(preds[str(j)])]))  # todo more for str
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def extract_features(extractor, data_loader):
    since = time.time()
    features = []
    targets = []
    for inputs, labels in data_loader:
        feats = extractor(inputs)
        features.append(feats)
        targets.append(labels)
    return features, targets


def main():
    plt.ion()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = get_params()
    data_dir = os.path.join(params.dataset_path, params.data_dir)
    split_file = os.path.join(params.dataset_path, params.split_file)

    train_form = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    training_set = WashingtonDataset(data_dir, split_file, params.data_type, params.split, mode='train',
                                     loader=pil_loader, transform=train_form)
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(training_set, batch_size, shuffle=True)

    test_form = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_set = WashingtonDataset(data_dir, split_file, params.data_type, params.split, mode='test',
                                 loader=pil_loader, transform=test_form)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)

    # for i, data_samples in enumerate(train_loader):
    #    img, target = data_samples

    # inputs, classes = next(iter(train_loader))

    # print(training_set.__len__())

    # inputs, classes = next(iter(train_loader))
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[wrgbd51.class_id_to_name[str(x)] for x in np.array(classes)])

    # load a pretrained model and reset final fully connected layer
    model_ft = models.alexnet(pretrained=True)
    extracted_layer = 1
    extractor = AlexNetExtractor(model_ft, extracted_layer)

    # train_features, train_labels = extract_features(extractor, train_loader)
    test_features, test_labels = extract_features(extractor, test_loader)

    print('train: {} {} /t test: {} {}'.format(  # train_features.__sizeof__(), train_labels.__sizeof__(),
                                               test_features.__sizeof__(), test_labels.__sizeof__()))

    # train_inputs, train_labels = next(iter(train_loader))
    #
    # train_features = extractor(train_inputs)
    #
    # test_inputs, test_labels = next(iter(test_loader))
    # test_features = extractor(test_inputs)
    #
    # device = torch.device("cpu")
    #
    # lin_clf = svm.LinearSVC()
    # lin_clf.fit(train_features.detach().numpy(), train_labels)
    # preds = lin_clf.predict(test_features.detach().numpy())
    # print('{}'.format(np.mean(preds == test_labels.numpy()) * 100))


if __name__ == '__main__':
    main()
    # print(wrgbd51.class_names.__len__())

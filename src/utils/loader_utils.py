import os

import h5py
import numpy as np
from PIL import Image

from basic_utils import RunSteps, DataTypes
from depth_utils import colorized_surfnorm


def cnn_or_rnn_features_loader(path):
    cnn_or_rnn_feats = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': [], 'layer5': [], 'layer6': [],
                        'layer7': []}
    img_file = h5py.File(path, 'r')
    for layer in cnn_or_rnn_feats.keys():
        cnn_or_rnn_feats[layer] = np.squeeze(np.asarray(img_file[layer]))

    return cnn_or_rnn_feats


def custom_loader(path, params):
    if params.data_type == DataTypes.Depth:
        results_dir = params.dataset_path + params.features_root + RunSteps.COLORIZED_DEPTH_SAVE + '/' + \
                      'all' + '_results_' + params.data_type
        if os.path.exists(results_dir):  # if colorized depth images are already saved read them
            img_path = results_dir + '/' + path.split('/')[-1] + '.hdf5'
            img_file = h5py.File(img_path, 'r')
            data_type = 'colorized_depth'
            return np.asarray(img_file[data_type])
        else:
            img = colorized_surfnorm(path)
            return np.array(img, dtype=np.float32)
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


def sunrgbd_loader(path, params):
    if params.data_type == "Depth_Colorized_HDF5":
        data_type = 'sunrgbd'
        img_file = h5py.File(path, 'r')
        return np.asarray(img_file[data_type], dtype=np.float32)
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

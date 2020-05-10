import fnmatch
import os

import numpy as np
import scipy.io as io
from torch.utils.data import Dataset

import wrgbd51
from basic_utils import RunSteps

"""
# WashingtonAll class is used to extract all cnn features at once without train/test splits.
# train/test splits are chosen later from the already saved/extracted files/features.
"""


class WashingtonAll(Dataset):
    def __init__(self, params, loader=None, transform=None):
        self.params = params
        self.loader = loader
        self.transform = transform
        self.data = self._init_dataset()

    def __getitem__(self, index):
        inp_path, out_path = self.data[index]
        datum = self.loader(inp_path, self.params)
        if self.transform is not None:
            datum = self.transform(datum)

        return datum, out_path

    def __len__(self):
        return len(self.data)

    def _init_dataset(self):
        data = []
        data_path = os.path.join(self.params.dataset_path, 'eval-set/')
        results_dir = self.params.dataset_path + self.params.features_root + self.params.proceed_step + '/' + \
                      self.params.net_model + '_results_' + self.params.data_type
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for category in sorted(os.listdir(data_path)):
            category_path = os.path.join(data_path, category)

            for instance in sorted(os.listdir(category_path)):
                instance_path = os.path.join(category_path, instance)

                data.extend(self.add_item(instance_path, results_dir))

        return data

    def add_item(self, instance_path, results_dir):
        indices = []
        suffix = '*_' + self.params.data_type + '.png'
        num_debug = 0

        for file in fnmatch.filter(sorted(os.listdir(instance_path)), suffix):
            path = os.path.join(instance_path, file)
            result_filename = results_dir + "/" + file + '.hdf5'
            item = (path, result_filename)
            indices.append(item)
            # get the first #debug_size (default=10) of sorted samples from each instance
            num_debug += 1
            if num_debug == self.params.debug_size and self.params.debug_mode:
                break
        return indices


"""
# WashingtonDataset class is for the use of Washington RGB-D dataset as is provided by the 10-fold train/test splits.
"""


class WashingtonDataset(Dataset):
    def __init__(self, params, phase, loader=None, transform=None):
        self.params = params
        self.phase = phase
        self.loader = loader
        self.transform = transform
        self.data = self._init_dataset()

    def __getitem__(self, index):
        path, target = self.data[index]
        # if we are loading from the already available cnn features, then the loader is @cnnfeatures_loader
        # otherwise, for "fine-tuning", the loader is @custom_loader
        if self.params.proceed_step in (RunSteps.FIX_RECURSIVE_NN, RunSteps.FINE_RECURSIVE_NN):
            datum = self.loader(path)
        else:   # fine-tuning
            datum = self.loader(path, self.params)

        if self.transform is not None:
            datum = self.transform(datum)

        filename = path[path.rfind('/') + 1:]

        return datum, target, filename

    def __len__(self):
        return len(self.data)

    def _init_dataset(self):
        data = []
        data_path = os.path.join(self.params.dataset_path, 'eval-set/')
        split_file = os.path.join(self.params.dataset_path, 'splits.mat')

        split_data = io.loadmat(split_file)['splits'].astype(np.uint8)
        test_instances = split_data[:, self.params.split_no - 1]

        for category in sorted(os.listdir(data_path)):

            category_path = os.path.join(data_path, category)
            cat_ind = int(wrgbd51.class_name_to_id[category])

            for instance in sorted(os.listdir(category_path)):
                instance_path = os.path.join(category_path, instance)

                if self.phase == 'test':
                    if test_instances[cat_ind] == np.uint8(instance.split('_')[-1]):
                        data.extend(self.add_item(instance_path, cat_ind))
                else:
                    if test_instances[cat_ind] != np.uint8(instance.split('_')[-1]):
                        data.extend(self.add_item(instance_path, cat_ind))

        return data

    def add_item(self, instance_path, cat_ind):
        indices = []
        suffix = '*_' + self.params.data_type + '.png'
        num_debug = 0

        for file in fnmatch.filter(sorted(os.listdir(instance_path)), suffix):
            if self.params.proceed_step == RunSteps.FIX_RECURSIVE_NN:
                # this means rnn features are already saved or wanted to be load
                if self.params.load_features:
                    rnn_feats_path = self.params.dataset_path + self.params.features_root + self.params.proceed_step + \
                                     '/' + self.params.net_model + '_results_' + self.params.data_type
                    path = rnn_feats_path + '/' + file + '.hdf5'
                else:
                    # the already extracted fixed CNN features are expected in the below path
                    cnn_feats_path = self.params.dataset_path + self.params.features_root + RunSteps.FIX_EXTRACTION + \
                                     '/' + self.params.net_model + '_results_' + self.params.data_type
                    path = cnn_feats_path + '/' + file + '.hdf5'
            elif self.params.proceed_step == RunSteps.FINE_RECURSIVE_NN:
                # this means rnn features are already saved or wanted to be load
                if self.params.load_features:
                    rnn_feats_path = self.params.dataset_path + self.params.features_root + self.params.proceed_step + \
                                     '/' + self.params.net_model + '_results_' + self.params.data_type + \
                                     '/split_' + str(self.params.split_no)
                    path = rnn_feats_path + '/' + file + '.hdf5'
                else:
                    # the already extracted finetuned CNN features are expected in the below path
                    cnn_feats_path = self.params.dataset_path + self.params.features_root + RunSteps.FINE_TUNING + \
                                     '/split-' + str(self.params.split_no) + '/' + self.params.net_model + \
                                     '_results_' + self.params.data_type
                    path = cnn_feats_path + '/' + file + '.hdf5'
            else:
                path = os.path.join(instance_path, file)

            item = (path, cat_ind)
            indices.append(item)
            # get the first debug_size (default=10) of sorted samples from each instance
            num_debug += 1
            if num_debug == self.params.debug_size and self.params.debug_mode:
                break
        return indices

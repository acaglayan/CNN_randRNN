import os
import numpy as np
import scipy.io as io
from torch.utils.data import Dataset

import sunrgbd
from basic_utils import RunSteps
from sunrgbd import load_props


class DataTypes:
    RGB = 'RGB_JPG'
    Depth = 'Depth_Colorized_HDF5'
    RGBD = 'RGBD'


class SUNRGBDDataset(Dataset):
    def __init__(self, params, phase, loader=None, transform=None):
        self.params = params
        self.phase = phase
        self.loader = loader
        self.transform = transform
        if params.proceed_step == RunSteps.SAVE_SUNRGBD:
            self.data = self._save_dataset()
        else:
            self.data = self._init_dataset()

    def __getitem__(self, index):
        path, label_id = self.data[index]

        datum = self.loader(path, self.params)

        if self.params.data_type == DataTypes.Depth:
            filename = path[path.rfind('/') + 1:-5]
        else:
            filename = path[path.rfind('/') + 1:]

        if self.transform is not None:
            datum = self.transform(datum)

        return datum, label_id, filename

    def __len__(self):
        return len(self.data)

    def _save_dataset(self):
        data = []
        split_file = os.path.join(self.params.dataset_path, 'allsplit.mat')
        sunrgbd_meta_file = os.path.join(self.params.dataset_path, 'SUNRGBDMeta.mat')
        sunrgbd_meta = io.loadmat(sunrgbd_meta_file)['SUNRGBDMeta']

        if self.phase == 'train':
            data_paths = io.loadmat(split_file)['alltrain'][0]
        else:
            data_paths = io.loadmat(split_file)['alltest'][0]

        num_debug = 0
        for path in data_paths:
            current_img = load_props(self.params, str(path), self.phase)
            if current_img.is_scene_challenge_category():
                sequence = 'SUNRGBD' + current_img.sequence_name

                if sequence[-1] == '/':
                    sequence = sequence[:-1]
                current_img.Rtilt = np.asarray(sunrgbd_meta[sunrgbd_meta['sequenceName'] == sequence]['Rtilt'][0],
                                               dtype=np.float32)
                current_img.K = np.asarray(sunrgbd_meta[sunrgbd_meta['sequenceName'] == sequence]['K'][0],
                                           dtype=np.float32)
                data.append(current_img)

                num_debug += 1
                if num_debug == self.params.debug_size and self.params.debug_mode:
                    break

        return data

    def _init_dataset(self):
        data = []
        data_path = self.params.dataset_path + RunSteps.SAVE_SUNRGBD + '/' + self.params.data_type + '/' + self.phase

        for file in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, file)
            label = file[:file.find('__')]
            label_id = np.int(sunrgbd.class_name_to_id[label])
            data.append((path, label_id))

        return data

import os
import shutil

import h5py
import numpy as np

from basic_utils import profile
from depth_utils import colorized_surfnorm_sunrgbd
from sunrgbd_loader import DataTypesSUNRGBD, SUNRGBDDataset


@profile
def process_dataset_save(params):
    data_type = 'sunrgbd'
    train_set = SUNRGBDDataset(params, phase='train')
    test_set = SUNRGBDDataset(params, phase='test')

    all_dataset = {'train': train_set, 'test': test_set}

    for phase in ['train', 'test']:
        results_dir = params.dataset_path + params.proceed_step + '/' + params.data_type + "/" + phase
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for sunrgbd_img in all_dataset[phase].data:
            result_filename = results_dir + '/' + sunrgbd_img.get_fullname()
            if params.data_type == DataTypesSUNRGBD.Depth:
                result_filename += '.hdf5'
                with h5py.File(result_filename, 'w') as f:
                    f.create_dataset(data_type,
                                     data=np.array(colorized_surfnorm_sunrgbd(sunrgbd_img), dtype=np.float32))
                f.close()
            else:
                shutil.copy(sunrgbd_img.path, result_filename)

import fnmatch
import os

import h5py
import numpy as np

from basic_utils import profile
from depth_utils import colorized_surfnorm


@profile
def process_depth_save(params):
    suffix = '*_' + params.data_type + '.png'
    data_path = os.path.join(params.dataset_path, 'eval-set/')
    results_dir = params.dataset_path + params.features_root + params.proceed_step + '/' + \
                  params.net_model + '_results_' + params.data_type
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    data_type = 'colorized_depth'
    for category in sorted(os.listdir(data_path)):
        category_path = os.path.join(data_path, category)

        for instance in sorted(os.listdir(category_path)):
            instance_path = os.path.join(category_path, instance)
            num_debug = 0

            for file in fnmatch.filter(sorted(os.listdir(instance_path)), suffix):
                path = os.path.join(instance_path, file)
                result_filename = results_dir + "/" + file + '.hdf5'

                with h5py.File(result_filename, 'w') as f:
                    f.create_dataset(data_type, data=np.array(colorized_surfnorm(path), dtype=np.float32))
                f.close()
                # get the first #debug_size (default=10) of sorted samples from each instance
                num_debug += 1
                if num_debug == params.debug_size and params.debug_mode:
                    break

import gc
import logging
import os
import time
import warnings
import numpy as np
import psutil
import torch
from sklearn import svm


def classify(train_data, train_labels, test_data):
    # LinearSVC parameters
    # dual : bool, (default=True)
    # Select the algorithm to either solve the dual or primal optimization problem.
    # Prefer dual=False when n_samples > n_features.
    # max_iter=2000, loss='hinge'
    clf = svm.LinearSVC()
    clf.fit(train_data, train_labels)

    preds = clf.predict(test_data)
    confidence_scores = clf.decision_function(test_data)

    return preds, confidence_scores


class Models:
    AlexNet = 'alexnet'
    VGGNet16 = 'vgg16_bn'
    ResNet50 = 'resnet50'
    ResNet101 = 'resnet101'
    DenseNet121 = 'densenet121'
    ALL = [AlexNet, VGGNet16, ResNet50, ResNet101, DenseNet121]


class DataTypes:
    RGB = 'crop'
    Depth = 'depthcrop'
    RGBD = 'rgbd'
    ALL = [RGB, Depth, RGBD]


class DataTypesSUNRGBD:
    RGB = 'RGB_JPG'
    Depth = 'Depth_Colorized_HDF5'
    RGBD = 'RGBD'


class Pools:
    MAX = 'max'
    AVG = 'avg'
    RANDOM = 'random'
    ALL = [MAX, AVG, RANDOM]


class OverallModes:
    FINETUNE_MODEL = 1
    FIX_PRETRAIN_MODEL = 2
    FUSION = 3
    ALL = [FINETUNE_MODEL, FIX_PRETRAIN_MODEL, FUSION]


class RunSteps:
    COLORIZED_DEPTH_SAVE = 'colorized_depth_images'
    FIX_EXTRACTION = 'fixed_extraction'
    FIX_RECURSIVE_NN = 'fixed_recursive_nn'
    FINE_TUNING = 'fine_tuning'
    FINE_EXTRACTION = 'finetuned_extraction'
    FINE_RECURSIVE_NN = 'finetuned_recursive_nn'
    OVERALL_RUN = 'overall_pipeline_run'
    SAVE_SUNRGBD = "organized-set"


class PrForm:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    END_FORMAT = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# reference for source code of profiling:
# https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python


def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        logging.info("{}: memory before: {:}, after: {:}, consumed: {:}; exec time: {}".format(
            func.__name__,
            format_bytes(mem_before), format_bytes(mem_after), format_bytes(mem_after - mem_before),
            elapsed_time))
        return result

    return wrapper


def format_bytes(mem_bytes):
    if abs(mem_bytes) < 1000:
        return str(mem_bytes) + "B"
    elif abs(mem_bytes) < 1e6:
        return str(round(mem_bytes / 1e3, 2)) + "kB"
    elif abs(mem_bytes) < 1e9:
        return str(round(mem_bytes / 1e6, 2)) + "MB"
    else:
        return str(round(mem_bytes / 1e9, 2)) + "GB"


def numpy2tensor(np_var, device=torch.device("cuda")):
    return torch.from_numpy(np_var.copy()).to(device)


def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()


def calc_mem(tensor_type, size):
    if tensor_type in [torch.float32, torch.float, torch.int32, torch.int]:
        byte_len = 4
    elif tensor_type in [torch.float64, torch.double, torch.int64, torch.long]:
        byte_len = 8
    elif tensor_type in [torch.float16, torch.half, torch.int16, torch.short]:
        byte_len = 2
    elif tensor_type in [torch.uint8, torch.int8, torch.bool]:
        byte_len = 1
    else:
        print('\n{}{}{}Tensor type is {} {}'.format(PrForm.RED, PrForm.UNDERLINE, PrForm.BOLD,
                                                    tensor_type, PrForm.END_FORMAT))
        raise NotImplementedError
    return byte_len * size


def report_gpu_memory():
    warnings.filterwarnings("ignore", category=UserWarning)
    total = 0
    print('{}{}GPU Stats{}'.format(PrForm.GREEN, PrForm.BOLD, PrForm.END_FORMAT))
    print('-' * 20)
    print('{}{}Element type\t\t\t\t\t\t\t\t\t\t\t\tSize\t\t\t\t\t\tUsed MEM{}'.format(PrForm.MAGENTA,
                                                                                      PrForm.BOLD,
                                                                                      PrForm.END_FORMAT))
    print('-' * 100)
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            mem = calc_mem(obj.dtype, np.prod(obj.shape))
            total += mem
            print('{: >24}\t\t\t{: >24}{: >24}'.format(str(type(obj)), str(obj.shape), format_bytes(mem)))
    print('-' * 100)
    print('{}{}Total Memory: {}{}{}\n'.format(PrForm.BLUE, PrForm.BOLD, PrForm.UNDERLINE,
                                              format_bytes(total), PrForm.END_FORMAT))


def report_cpu_stats():
    print('{}{}CPU Stats{}'.format(PrForm.GREEN, PrForm.BOLD, PrForm.END_FORMAT))
    print('-' * 20)
    print('CPU Percent: {}'.format(psutil.cpu_percent()))
    print('Virtual Mem: {}'.format(psutil.virtual_memory()))
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = format_bytes(py.memory_info()[0])
    print('Memory : {}\n'.format(memory_use))

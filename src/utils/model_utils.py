import numpy as np
from torchvision import transforms

import depth_transform
from basic_utils import DataTypes


def init_random_weights(num_split, chunk_size, rfs, opt):
    if opt == 'reduce_rfs':
        num_map = chunk_size
        rfs = int(np.sqrt(num_split)) * rfs[0]
        rfs = (rfs, rfs)
    else:
        num_map = chunk_size * num_split

    weights = np.zeros(shape=(num_map,) + rfs, dtype=np.float32)
    for i in range(num_map):
        random_weight = -0.1 + 0.2 * np.random.rand(rfs[0], rfs[1])
        weights[i, :] = random_weight

    return weights


def randomized_pool(weights, layer_inp, num_split):
    assert layer_inp.ndim == 4
    num_maps = layer_inp.shape[1]
    assert np.mod(num_maps, num_split) < 1e-15
    chunk_size = int(num_maps / num_split)
    rfs = (layer_inp.shape[2], layer_inp.shape[3])

    out_layer = np.multiply(layer_inp, weights)
    out_layer = np.reshape(out_layer, (out_layer.shape[0], num_split, chunk_size,) + rfs)
    out_layer = np.sum(out_layer, axis=1)

    return out_layer


def avg_pool(layer_inp, num_split):
    assert layer_inp.ndim == 4
    num_maps = layer_inp.shape[1]
    assert np.mod(num_maps, num_split) < 1e-15
    chunk_size = int(num_maps / num_split)

    layer_inp = np.reshape(layer_inp,
                           (layer_inp.shape[0], num_split, chunk_size, layer_inp.shape[2], layer_inp.shape[3]))

    out = np.mean(layer_inp, axis=1)

    return out


def max_pool(layer_inp, num_split):
    assert layer_inp.ndim == 4
    num_maps = layer_inp.shape[1]
    assert np.mod(num_maps, num_split) < 1e-15
    chunk_size = int(num_maps / num_split)

    layer_inp = np.reshape(layer_inp,
                           (layer_inp.shape[0], num_split, chunk_size, layer_inp.shape[2], layer_inp.shape[3]))

    out = np.max(layer_inp, axis=1)

    return out


def get_data_transform(data_type):
    std = [0.229, 0.224, 0.225]

    if data_type == DataTypes.RGB:
        mean = [0.485, 0.456, 0.406]
        data_form = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        mean = [0.0, 0.0, 0.0]  # [0.485, 0.456, 0.406]
        data_form = depth_transform.Compose([
            depth_transform.Resize(size=(256, 256), interpolation='NEAREST'),
            depth_transform.CenterCrop(224),
            depth_transform.ToTensor(),
            depth_transform.Normalize(mean, std)
        ])

    return data_form


def reshape_4d(layer_feats, shape):
    layer_feats = np.reshape(layer_feats, (layer_feats.shape[0],) + shape)

    return layer_feats


def flat_2d(data):
    return np.reshape(data, (data.shape[0], -1))


def get_num_maps(data4d):
    return data4d.shape[1]


def get_rfs(data4d):
    return data4d.shape[2], data4d.shape[3]

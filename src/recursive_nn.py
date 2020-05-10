import numpy as np


def forward(data, weights, num_rnn, rfs):
    data = np.transpose(data, (1, 2, 3, 0))
    num_maps, rows, cols, num_imgs = np.shape(data)
    assert rows == cols
    depth = np.floor(np.log(rows) / np.log(rfs[0]) + 0.5)
    depth = depth.astype(int)

    # ensure a balanced tree is possible with these sizes
    assert np.mod(np.log(rows) / np.log(rfs[0]), 1) < 1e-15
    assert np.mod(np.log(cols) / np.log(rfs[1]), 1) < 1e-15

    rnn_data = np.zeros(shape=(num_rnn, num_maps, num_imgs), dtype=np.float32)
    for r in range(0, num_rnn):
        if np.mod(r + 1, 8) == 0:
            print('RNN: {}'.format(r + 1))
        w = np.squeeze(weights[r, :, :])
        tree = data
        for layer in range(0, depth):
            new_tree = np.zeros(shape=(num_maps, int(tree.shape[1]/rfs[0]), int(tree.shape[2]/rfs[1]), num_imgs),
                                dtype=np.float32)
            rc = 0
            for row in range(0, tree.shape[1], rfs[0]):
                cc = 0
                for col in range(0, tree.shape[2], rfs[1]):
                    curr_data = tree[:, row:row + rfs[0], col:col + rfs[1], :]
                    child = np.dot(w, curr_data.reshape(-1, num_imgs))
                    new_tree[:, rc, cc, :] = np.tanh(child)
                    cc += 1
                rc += 1
            tree = new_tree
        rnn_data[r, :, :] = np.reshape(np.squeeze(tree), (-1, num_imgs))

    rnn_data = np.transpose(rnn_data, [2, 0, 1])
    return rnn_data


def init_random_weights(num_rnn, inp_shape):
    num_maps = inp_shape[0]
    rfs = inp_shape[1:3]
    prod_rfs = np.prod(rfs)

    weights = np.zeros(shape=(num_rnn, num_maps, num_maps * prod_rfs), dtype=np.float32)
    for i in range(0, num_rnn):
        weights[i, :, :] = -0.1 + 0.2 * np.random.rand(num_maps, num_maps * prod_rfs)
    return weights


def forward_rnn(weights, data, num_rnn, inp_shape):
    rfs = inp_shape[1:3]
    print('RNN forward propagation through the data..')
    rnn_data = forward(data, weights, num_rnn, rfs)
    data_samples = data.shape[0]
    rnn_data = np.reshape(rnn_data, (data_samples, -1))

    return rnn_data


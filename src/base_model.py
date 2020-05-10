import logging
import abc
import pickle
import h5py
import os
import numpy as np
import torch
import recursive_nn
from basic_utils import PrForm, DataTypes, RunSteps, Pools
from model_utils import avg_pool, max_pool, randomized_pool
from utils import model_utils
import basic_utils
from loader_utils import cnn_or_rnn_features_loader
from wrgbd_loader import WashingtonDataset


def reduce_rfs(weights, layer_feats, num_reducing, pool_method):
    # check the size availability
    assert np.mod(layer_feats.shape[2], np.sqrt(num_reducing)) < 1e-15
    weight_len = int(np.sqrt(num_reducing))

    if pool_method == Pools.AVG:
        t_avg_pool = torch.nn.AvgPool2d(kernel_size=weight_len, stride=weight_len)
        result = t_avg_pool(basic_utils.numpy2tensor(layer_feats, device=torch.device("cpu")))
    elif pool_method == Pools.MAX:
        t_max_pool = torch.nn.MaxPool2d(kernel_size=weight_len, stride=weight_len)
        result = t_max_pool(basic_utils.numpy2tensor(layer_feats, device=torch.device("cpu")))
    else:  # Pools.RANDOM
        result = np.multiply(layer_feats, weights)
        t_avg_pool = torch.nn.AvgPool2d(kernel_size=weight_len, stride=weight_len)
        result = t_avg_pool(basic_utils.numpy2tensor(result, device=torch.device("cpu"))) * num_reducing

    return basic_utils.tensor2numpy(result)


def reduce_map(weights, layer_feats, num_split, pool_method):
    if pool_method == Pools.AVG:
        train_inp = avg_pool(layer_feats, num_split=num_split)
    elif pool_method == Pools.MAX:
        train_inp = max_pool(layer_feats, num_split=num_split)
    else:  # pool_method is random
        train_inp = randomized_pool(weights, layer_feats, num_split=num_split)

    return train_inp


'''
pool_method: pooling method while proceeding reduce operation. 'avg', 'max', and 'random' are 
             choices.
opt: option for reduce operation. 'reduce_map' and 'reduce_rfs' are defined choices.
'''


def reduce_inp(weights, layer_feats, num_split, pool_method, opt):
    if opt == 'reduce_rfs':
        rnn_inp = reduce_rfs(weights, layer_feats, num_split, pool_method)
    else:
        rnn_inp = reduce_map(weights, layer_feats, num_split, pool_method)

    return rnn_inp


class Model:

    def __init__(self, params):
        self.params = params
        self.train_labels, self.test_labels = [], []
        self.rnn_train_outs = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': [],
            'layer5': [],
            'layer6': [],
            'layer7': []
        }
        self.rnn_test_outs = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': [],
            'layer5': [],
            'layer6': [],
            'layer7': []
        }
        if not params.load_features:
            self.reduction_weights = self.reduction_random_weights()
            self.rnn_weights = self.rnn_random_weights()

    def convert_rnn_features(self):
        for layer in self.rnn_train_outs.keys():
            self.rnn_train_outs[layer] = np.concatenate([np.array(i) for i in self.rnn_train_outs[layer]])
            self.rnn_test_outs[layer] = np.concatenate([np.array(i) for i in self.rnn_test_outs[layer]])

    def convert_labels(self):
        self.train_labels = np.concatenate([np.array(i) for i in self.train_labels])
        self.test_labels = np.concatenate([np.array(i) for i in self.test_labels])

    @abc.abstractmethod
    def model_structure(self):
        """:return rnn input shapes for each layer"""

    @abc.abstractmethod
    def model_reduction_plan(self):
        """defines layer wise reduction plan for each model"""

    @abc.abstractmethod
    def process_layer1(self, curr_inputs):
        """this method pre-process layer1 cnn features in each net model according to its inputs before rnn.
        :return processed input :param curr_inputs is current cnn features taken from batch"""

    @abc.abstractmethod
    def process_layer2(self, curr_inputs):
        """this method pre-process layer2 cnn features in each net model according to its inputs before rnn.
        :return processed input :param curr_inputs is current cnn features taken from batch"""

    @abc.abstractmethod
    def process_layer3(self, curr_inputs):
        """this method pre-process layer3 cnn features in each net model according to its inputs before rnn.
        :return processed input :param curr_inputs is current cnn features taken from batch"""

    @abc.abstractmethod
    def process_layer4(self, curr_inputs):
        """this method pre-process layer4 cnn features in each net model according to its inputs before rnn.
        :return processed input :param curr_inputs is current cnn features taken from batch"""

    @abc.abstractmethod
    def process_layer5(self, curr_inputs):
        """this method pre-process layer5 cnn features in each net model according to its inputs before rnn.
        :return processed input :param curr_inputs is current cnn features taken from batch"""

    @abc.abstractmethod
    def process_layer6(self, curr_inputs):
        """this method pre-process layer6 cnn features in each net model according to its inputs before rnn.
        :return processed input :param curr_inputs is current cnn features taken from batch"""

    @abc.abstractmethod
    def process_layer7(self, curr_inputs):
        """this method pre-process layer7 cnn features in each net model according to its inputs before rnn.
        :return processed input :param curr_inputs is current cnn features taken from batch"""

    def calc_scores(self, preds):
        result = (preds == self.test_labels)
        avg_res = np.mean(result) * 100
        true_preds = np.count_nonzero(result == True)
        test_size = np.size(result)
        return avg_res, true_preds, test_size

    def classify_cnn_features(self, layer_train, layer_test):
        if layer_train.ndim == 4:
            layer_train = model_utils.flat_2d(layer_train)
            layer_test = model_utils.flat_2d(layer_test)
        logging.info('CNN feature dimension {}'.format(layer_train.shape[1]))
        preds, confidence_scores = basic_utils.classify(layer_train, self.train_labels, layer_test)

        avg_res_cnn, true_preds_cnn, test_size_cnn = self.calc_scores(preds)

        logging.info('CNN result: {0:.2f}% ({1}/{2})..'.format(avg_res_cnn, true_preds_cnn, test_size_cnn))

    def classify_rnn_features(self, layer_train, layer_test):
        logging.info('RNN feature dimension {}'.format(layer_train.shape[1]))
        preds, confidence_scores = basic_utils.classify(layer_train, self.train_labels, layer_test)

        avg_res_rnn, true_preds_rnn, test_size_rnn = self.calc_scores(preds)

        logging.info('RNN result: {0:.2f}% ({1}/{2})..'.format(avg_res_rnn, true_preds_rnn, test_size_rnn))
        return confidence_scores

    def eval_layer1(self):
        curr_layer = 'layer1'
        logging.info('Running Layer-1...')
        logging.info('RNN with {} shapes. Preprocessed: {}'.format(self.model_structure()[curr_layer],
                                                                   self.model_reduction_plan()[curr_layer]))
        return self.classify_rnn_features(self.rnn_train_outs[curr_layer], self.rnn_test_outs[curr_layer])

    def eval_layer2(self):
        curr_layer = 'layer2'
        logging.info('Running Layer-2...')
        logging.info('RNN with {} shapes. Preprocessed: {}'.format(self.model_structure()[curr_layer],
                                                                   self.model_reduction_plan()[curr_layer]))
        return self.classify_rnn_features(self.rnn_train_outs[curr_layer], self.rnn_test_outs[curr_layer])

    def eval_layer3(self):
        curr_layer = 'layer3'
        logging.info('Running Layer-3...')
        logging.info('RNN with {} shapes. Preprocessed: {}'.format(self.model_structure()[curr_layer],
                                                                   self.model_reduction_plan()[curr_layer]))
        return self.classify_rnn_features(self.rnn_train_outs[curr_layer], self.rnn_test_outs[curr_layer])

    def eval_layer4(self):
        curr_layer = 'layer4'
        logging.info('Running Layer-4...')
        logging.info('RNN with {} shapes. Preprocessed: {}'.format(self.model_structure()[curr_layer],
                                                                   self.model_reduction_plan()[curr_layer]))
        return self.classify_rnn_features(self.rnn_train_outs[curr_layer], self.rnn_test_outs[curr_layer])

    def eval_layer5(self):
        curr_layer = 'layer5'
        logging.info('Running Layer-5...')
        logging.info('RNN with {} shapes. Preprocessed: {}'.format(self.model_structure()[curr_layer],
                                                                   self.model_reduction_plan()[curr_layer]))
        return self.classify_rnn_features(self.rnn_train_outs[curr_layer], self.rnn_test_outs[curr_layer])

    def eval_layer6(self):
        curr_layer = 'layer6'
        logging.info('Running Layer-6...')
        logging.info('RNN with {} shapes. Preprocessed: {}'.format(self.model_structure()[curr_layer],
                                                                   self.model_reduction_plan()[curr_layer]))
        return self.classify_rnn_features(self.rnn_train_outs[curr_layer], self.rnn_test_outs[curr_layer])

    def eval_layer7(self):
        curr_layer = 'layer7'
        logging.info('Running Layer-7...')
        logging.info('RNN with {} shapes. Preprocessed: {}'.format(self.model_structure()[curr_layer],
                                                                   self.model_reduction_plan()[curr_layer]))
        return self.classify_rnn_features(self.rnn_train_outs[curr_layer], self.rnn_test_outs[curr_layer])

    @basic_utils.profile
    def eval(self):
        if not self.params.load_features:
            logging.info('----------\n')

        training_set = WashingtonDataset(self.params, phase='train', loader=cnn_or_rnn_features_loader)
        train_loader = torch.utils.data.DataLoader(training_set, self.params.batch_size, shuffle=False)

        test_set = WashingtonDataset(self.params, phase='test', loader=cnn_or_rnn_features_loader)
        test_loader = torch.utils.data.DataLoader(test_set, self.params.batch_size, shuffle=False)

        for phase_loader in [train_loader, test_loader]:
            batch_ind = -1
            for inputs, labels, filenames in phase_loader:
                batch_ind += 1
                for layer in self.model_structure().keys():
                    curr_layer_inp = inputs[layer].numpy()

                    if self.params.load_features:
                        curr_rnn_out = curr_layer_inp
                    else:
                        curr_layer_inp = self.process_layer(layer, curr_layer_inp)
                        curr_rnn_out = recursive_nn.forward_rnn(self.rnn_weights[layer], curr_layer_inp,
                                                                self.params.num_rnn, self.model_structure()[layer])

                    if phase_loader is train_loader:
                        self.rnn_train_outs[layer].append(curr_rnn_out)
                    else:
                        self.rnn_test_outs[layer].append(curr_rnn_out)

                curr_labels = labels.numpy()
                if phase_loader == train_loader:
                    self.train_labels.append(curr_labels)
                else:
                    self.test_labels.append(curr_labels)

                if self.params.save_features:
                    self.save_recursive_features(filenames, batch_ind, phase=phase_loader.dataset.phase)

        self.convert_variables()

        if not self.params.fusion_levels:
            l1_conf_scores = self.eval_layer1()
            logging.info('----------\n')

            l2_conf_scores = self.eval_layer2()
            logging.info('----------\n')

            l3_conf_scores = self.eval_layer3()
            logging.info('----------\n')

            l4_conf_scores = self.eval_layer4()
            logging.info('----------\n')

            l5_conf_scores = self.eval_layer5()
            logging.info('----------\n')

            l6_conf_scores = self.eval_layer6()
            logging.info('----------\n')

            l7_conf_scores = self.eval_layer7()
            logging.info('----------\n')

            self.save_svm_conf_scores(l1_conf_scores, l2_conf_scores, l3_conf_scores, l4_conf_scores,
                                      l5_conf_scores, l6_conf_scores, l7_conf_scores)

        self.fusion_layers()
        logging.info('----------\n')

    def convert_variables(self):
        self.convert_rnn_features()
        self.convert_labels()

    def generate_reduction_randoms(self):
        all_weights = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': [],
            'layer5': [],
            'layer6': [],
            'layer7': []
        }
        model_reduction = self.model_reduction_plan()
        for layer in model_reduction.keys():
            weight = None
            for ind in range(0, len(model_reduction[layer])):
                num_split, chunk_size, rfs, opt = model_reduction[layer][ind]
                if num_split != 1:
                    weight = model_utils.init_random_weights(num_split, chunk_size, (rfs, rfs), opt)

                all_weights[layer].append(weight)
        return all_weights

    @basic_utils.profile
    def reduction_random_weights(self):
        if self.params.reuse_randoms:
            save_load_dir = self.params.dataset_path + self.params.features_root + 'random_weights/'
            reduc_weights_file = save_load_dir + self.params.net_model + '_reduction_random_weights.pkl'
            if not os.path.exists(save_load_dir):
                os.makedirs(save_load_dir)

            try:
                with open(reduc_weights_file, 'rb') as f:
                    all_weights = pickle.load(f)
                    return all_weights
            except Exception:
                print('{}{}Failed to load the reduction weights file! They are going to be created for the first '
                      'time!{} '.format(PrForm.YELLOW, PrForm.BOLD, PrForm.END_FORMAT))
                logging.info('The reduction weights are going to be saved into {}'.format(reduc_weights_file))
                all_weights = self.generate_reduction_randoms()
                with open(reduc_weights_file, 'wb') as f:
                    pickle.dump(all_weights, f, pickle.HIGHEST_PROTOCOL)
                return all_weights
            finally:
                f.close()
        else:
            return self.generate_reduction_randoms()

    def generate_rnn_randoms(self):
        rnn_all_layer_weights = {}
        model_structure = self.model_structure()
        for layer in model_structure.keys():
            weights = recursive_nn.init_random_weights(self.params.num_rnn, model_structure[layer])
            rnn_all_layer_weights[layer] = weights

        return rnn_all_layer_weights

    @basic_utils.profile
    def rnn_random_weights(self):
        if self.params.reuse_randoms:
            save_load_dir = self.params.dataset_path + self.params.features_root + 'random_weights/'
            rnn_weights_file = save_load_dir + self.params.net_model + '_rnn_random_weights.pkl'
            if not os.path.exists(save_load_dir):
                os.makedirs(save_load_dir)
            try:
                with open(rnn_weights_file, 'rb') as f:
                    rnn_all_weights = pickle.load(f)
                    return rnn_all_weights
            except Exception:
                print('{}{}Failed to load the RNN weights file! They are going to be created for the first time!{}'.
                      format(PrForm.YELLOW, PrForm.BOLD, PrForm.END_FORMAT))
                logging.info('The RNN weights are going to be saved into {}'.format(rnn_weights_file))
                rnn_all_weights = self.generate_rnn_randoms()
                with open(rnn_weights_file, 'wb') as f:
                    pickle.dump(rnn_all_weights, f, pickle.HIGHEST_PROTOCOL)
                return rnn_all_weights
            finally:
                f.close()
        else:
            return self.generate_rnn_randoms()

    def process_layer(self, layer, curr_layer_inp):
        if layer == 'layer1':
            processed_inp = self.process_layer1(curr_layer_inp)
        elif layer == 'layer2':
            processed_inp = self.process_layer2(curr_layer_inp)
        elif layer == 'layer3':
            processed_inp = self.process_layer3(curr_layer_inp)
        elif layer == 'layer4':
            processed_inp = self.process_layer4(curr_layer_inp)
        elif layer == 'layer5':
            processed_inp = self.process_layer5(curr_layer_inp)
        elif layer == 'layer6':
            processed_inp = self.process_layer6(curr_layer_inp)
        else:
            processed_inp = self.process_layer7(curr_layer_inp)

        return processed_inp

    def save_recursive_features(self, filenames, batch_ind, phase):

        save_dir = self.params.dataset_path + self.params.features_root + self.params.proceed_step + '/' + \
                   self.params.net_model + '_results_' + self.params.data_type

        if self.params.proceed_step == RunSteps.FINE_RECURSIVE_NN:
            save_dir += '/split_' + str(self.params.split_no)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(len(filenames)):
            path = save_dir + '/' + filenames[i]
            if '.hdf5' not in filenames[i]:
                path = path + '.hdf5'

            with h5py.File(path, 'w') as f:
                for extracted_layer in range(1, 8):
                    feature_type = 'layer' + str(extracted_layer)
                    if phase == 'train':
                        f.create_dataset(feature_type, data=self.rnn_train_outs[feature_type][batch_ind][i,])
                    else:
                        f.create_dataset(feature_type, data=self.rnn_test_outs[feature_type][batch_ind][i,])

                if phase == 'train':
                    f.create_dataset('labels', data=self.train_labels[batch_ind][i])
                else:
                    f.create_dataset('labels', data=self.test_labels[batch_ind][i])

            f.close()

    @abc.abstractmethod
    def get_best_trio_layers(self):
        """this method returns the best three layer of a model.
        :return l1, l2, l3 : best three consecutive layers """

    @abc.abstractmethod
    def get_best_modality_layers(self):
        """this method returns the best layers for each RGB and Depth modality for a model.
        :return rgb_best, depth_best : best layers for RGB and depth respectively """

    def save_svm_conf_scores(self, l1_conf_scores, l2_conf_scores, l3_conf_scores, l4_conf_scores, l5_conf_scores,
                             l6_conf_scores, l7_conf_scores):
        save_load_dir = self.params.dataset_path + self.params.features_root + self.params.proceed_step + \
                        '/svm_confidence_scores/'
        confidence_scores_file = save_load_dir + self.params.net_model + '_' + self.params.data_type + '_split_' + \
                                 str(self.params.split_no) + '.hdf5'

        if not os.path.exists(save_load_dir):
            os.makedirs(save_load_dir)

        with h5py.File(confidence_scores_file, 'w') as f:
            f.create_dataset('layer1', data=l1_conf_scores)
            f.create_dataset('layer2', data=l2_conf_scores)
            f.create_dataset('layer3', data=l3_conf_scores)
            f.create_dataset('layer4', data=l4_conf_scores)
            f.create_dataset('layer5', data=l5_conf_scores)
            f.create_dataset('layer6', data=l6_conf_scores)
            f.create_dataset('layer7', data=l7_conf_scores)
            f.create_dataset('labels', data=self.test_labels)
        f.close()

    def layer_concats(self):
        l1, l2, l3 = self.get_best_trio_layers()

        logging.info('Running Layer-[{}+{}] Feature Concat...'.format(l1, l2))
        logging.info('RNN features of {} and {} are concatenated'.format(l1, l2))
        self.classify_rnn_features(np.concatenate((self.rnn_train_outs[l1], self.rnn_train_outs[l2]), axis=1),
                                   np.concatenate((self.rnn_test_outs[l1], self.rnn_test_outs[l2]), axis=1))
        logging.info('----------\n')

        logging.info('Running Layer-[{}+{}] Feature Concat...'.format(l1, l3))
        logging.info('RNN features of {} and {} are concatenated'.format(l1, l3))
        self.classify_rnn_features(np.concatenate((self.rnn_train_outs[l1], self.rnn_train_outs[l3]), axis=1),
                                   np.concatenate((self.rnn_test_outs[l1], self.rnn_test_outs[l3]), axis=1))
        logging.info('----------\n')

        logging.info('Running Layer-[{}+{}]. Feature Concat..'.format(l2, l3))
        logging.info('RNN features of {} and {} are concatenated'.format(l2, l3))
        self.classify_rnn_features(np.concatenate((self.rnn_train_outs[l2], self.rnn_train_outs[l3]), axis=1),
                                   np.concatenate((self.rnn_test_outs[l2], self.rnn_test_outs[l3]), axis=1))
        logging.info('----------\n')

        logging.info('Running Layer-[5+6+7] Feature Concat...')
        logging.info('RNN features of {}, {} and {} are concatenated'.format(l1, l2, l3))
        self.classify_rnn_features(
            np.concatenate((self.rnn_train_outs[l1], self.rnn_train_outs[l2], self.rnn_train_outs[l3]),
                           axis=1),
            np.concatenate((self.rnn_test_outs[l1], self.rnn_test_outs[l2], self.rnn_test_outs[l3]),
                           axis=1)
        )

    def confidence_fusion(self):
        l1, l2, l3 = self.get_best_trio_layers()

        save_load_dir = self.params.dataset_path + self.params.features_root + self.params.proceed_step + \
                        '/svm_confidence_scores/'
        confidence_scores_file = save_load_dir + self.params.net_model + '_' + self.params.data_type + '_split_' + \
                                 str(self.params.split_no) + '.hdf5'
        try:
            with h5py.File(confidence_scores_file, 'r') as f:
                l1_conf_scores = np.asarray(f[l1])
                l2_conf_scores = np.asarray(f[l2])
                l3_conf_scores = np.asarray(f[l3])
                self.test_labels = np.asarray(f['labels'])
            f.close()
        except Exception as e:
            print('{}{}Failed to load the SVM confidence scores: {}{}'.format(PrForm.BOLD, PrForm.RED, e,
                                                                        PrForm.END_FORMAT))
            return

        ##### mean fusions
        """logging.info('Running Layer-[{}+{}] Confidence Average Fusion...'.format(l1, l2))
        logging.info('SVM confidence scores of {} and {} are average fused'.format(l1, l2))
        logging.info('SVM confidence average fusion')
        l12_avr_confidence = np.mean(np.array([l1_conf_scores, l2_conf_scores]), axis=0)
        l12_preds = np.argmax(l12_avr_confidence, axis=1)
        avg_res, true_preds, test_size = self.calc_scores(l12_preds)
        logging.info('Fusion result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))
        logging.info('----------\n')

        logging.info('Running Layer-[{}+{}] Confidence Average Fusion...'.format(l1, l3))
        logging.info('SVM confidence scores of {} and {} are average fused'.format(l1, l3))
        logging.info('SVM confidence average fusion')
        l13_avr_confidence = np.mean(np.array([l1_conf_scores, l3_conf_scores]), axis=0)
        l13_preds = np.argmax(l13_avr_confidence, axis=1)
        avg_res, true_preds, test_size = self.calc_scores(l13_preds)
        logging.info('Fusion result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))
        logging.info('----------\n')

        logging.info('Running Layer-[{}+{}] Confidence Average Fusion...'.format(l2, l3))
        logging.info('SVM confidence scores of {} and {} are average fused'.format(l2, l3))
        logging.info('SVM confidence average fusion')
        l23 = np.mean(np.array([l2_conf_scores, l3_conf_scores]), axis=0)
        l23_preds = np.argmax(l23, axis=1)
        avg_res, true_preds, test_size = self.calc_scores(l23_preds)
        logging.info('Fusion result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))
        logging.info('----------\n')"""

        logging.info('Running Layer-[{}+{}+{}] Confidence Average Fusion...'.format(l1, l2, l3))
        logging.info('SVM confidence scores of {}, {} and {} are average fused'.format(l1, l2, l3))
        logging.info('SVM confidence average fusion')
        l123_avr_confidence = np.mean(np.array([l1_conf_scores, l2_conf_scores, l3_conf_scores]), axis=0)
        l123_preds = np.argmax(l123_avr_confidence, axis=1)
        avg_res, true_preds, test_size = self.calc_scores(l123_preds)
        logging.info('Fusion result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))

    def fusion_layers(self):
        self.confidence_fusion()
        # logging.info('----------\n')
        # self.layer_concats()

    def calc_modality_weights(self, conf_scores):
        assert len(conf_scores) == 2
        l1_conf_scores = conf_scores[0]
        l2_conf_scores = conf_scores[1]
        # thresh = 0.0

        s_l1 = (np.sum(np.square(l1_conf_scores), axis=1))
        s_l2 = (np.sum(np.square(l2_conf_scores), axis=1))
        # s_l1[np.max(l1_conf_scores) < thresh] = np.finfo(np.float32).eps
        # s_l2[np.max(l2_conf_scores) < thresh] = np.finfo(np.float32).eps

        m_l1 = s_l1 / np.maximum(s_l1, s_l2)
        m_l2 = s_l2 / np.maximum(s_l1, s_l2)

        w_l1 = np.sqrt(np.exp(m_l1) / (np.exp(m_l1) + np.exp(m_l2)))
        w_l2 = 1 - w_l1

        return w_l1, w_l2

    def combine_rgbd(self):
        if self.params.proceed_step == RunSteps.OVERALL_RUN:
            self.params.data_type = DataTypes.RGB
            self.params.proceed_step = RunSteps.FIX_RECURSIVE_NN    # we take fix RGB results for RGB-D fusions
            l_rgb1, l_rgb2, l_rgb3 = self.get_best_trio_layers()
            rgb_best, _ = self.get_best_modality_layers()

            self.params.data_type = DataTypes.Depth
            self.params.proceed_step = RunSteps.FINE_RECURSIVE_NN  # we take finetuned Depth results for RGB-D fusions
            l_depth1, l_depth2, l_depth3 = self.get_best_trio_layers()
            _, depth_best = self.get_best_modality_layers()

            self.params.proceed_step = RunSteps.OVERALL_RUN
        else:
            self.params.data_type = DataTypes.RGB
            l_rgb1, l_rgb2, l_rgb3 = self.get_best_trio_layers()
            rgb_best, _ = self.get_best_modality_layers()

            self.params.data_type = DataTypes.Depth
            l_depth1, l_depth2, l_depth3 = self.get_best_trio_layers()
            _, depth_best = self.get_best_modality_layers()

        self.params.data_type = DataTypes.RGBD
        save_load_dir = self.params.dataset_path + self.params.features_root + self.params.proceed_step + \
                        '/svm_confidence_scores/'
        rgb_confidence_scores_file = save_load_dir + self.params.net_model + '_' + DataTypes.RGB + '_split_' + \
                                     str(self.params.split_no) + '.hdf5'
        depth_confidence_scores_file = save_load_dir + self.params.net_model + '_' + DataTypes.Depth + '_split_' + \
                                       str(self.params.split_no) + '.hdf5'

        try:
            rgb_scores_file = h5py.File(rgb_confidence_scores_file, 'r')
            depth_scores_file = h5py.File(depth_confidence_scores_file, 'r')

            rgb_scores = {l_rgb1: [], l_rgb2: [], l_rgb3: [], 'labels': []}
            depth_scores = {l_depth1: [], l_depth2: [], l_depth3: [], 'labels': []}
            for layer in rgb_scores.keys():
                rgb_scores[layer] = np.squeeze(np.asarray(rgb_scores_file[layer]))

            for layer in depth_scores.keys():
                depth_scores[layer] = np.squeeze(np.asarray(depth_scores_file[layer]))

            self.test_labels = rgb_scores['labels']

            rgb_best_score = np.squeeze(np.asarray(rgb_scores_file[rgb_best]))
            depth_best_score = np.squeeze(np.asarray(depth_scores_file[depth_best]))
        except Exception as e:
            print('{}{}Failed to load the SVM confidence scores: {}{}'.format(PrForm.BOLD, PrForm.RED, e,
                                                                        PrForm.END_FORMAT))
            return

        # logging.info('----------\n')
        # self.concat_rgbd()
        self.combine_one__bests(rgb_best_score, depth_best_score, rgb_best, depth_best)
        logging.info('----------\n')

        logging.info('Running Layer-[RGB({}+{}+{})+Depth({}+{}+{})] Average of Confidence Fusion for RGBD '
                     'Avr(Avr(rgb123), Avr(depth123))...'.format(l_rgb1, l_rgb2, l_rgb3, l_depth1, l_depth2, l_depth3))
        logging.info('Average SVM confidence scores of [RGB({}+{}+{})+Depth({}+{}+{})] are taken')
        logging.info('SVMs confidence average fusion for combined RGB and Depth')
        rgb_l123_avg_confidence = np.sum(np.array([rgb_scores[l_rgb1], rgb_scores[l_rgb2], rgb_scores[l_rgb3]]),
                                         axis=0)
        depth_l123_avg_confidence = np.sum(np.array([depth_scores[l_depth1], depth_scores[l_depth2],
                                                     depth_scores[l_depth3]]), axis=0)
        rgbd_l123_comb_confidence = np.mean(np.array([rgb_l123_avg_confidence, depth_l123_avg_confidence]), axis=0)
        l123_preds = np.argmax(rgbd_l123_comb_confidence, axis=1)
        avg_res, true_preds, test_size = self.calc_scores(l123_preds)
        logging.info('Combined Confidence Avg result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))
        logging.info('----------\n')

        logging.info('Running Layer-[RGB({}+{}+{})+Depth({}+{}+{})] Weighted of Confidence Fusion for RGBD '
                     'Weighted(Avr(rgb123), Avr(depth123))...'.format(l_rgb1, l_rgb2, l_rgb3, l_depth1, l_depth2,
                                                                      l_depth3))
        logging.info('Weighted Average SVM confidence scores of [RGB({}+{}+{})+Depth({}+{}+{})] are taken')
        logging.info('SVMs confidence weighted fusion')
        w_rgb, w_depth = self.calc_modality_weights((rgb_l123_avg_confidence, depth_l123_avg_confidence))
        rgbd_l123_wadd_confidence = np.add(rgb_l123_avg_confidence * w_rgb[:, np.newaxis],
                                           depth_l123_avg_confidence * w_depth[:, np.newaxis])
        l123_preds = np.argmax(rgbd_l123_wadd_confidence, axis=1)
        avg_res, true_preds, test_size = self.calc_scores(l123_preds)
        logging.info('Combined Weighted Confidence result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))

    def concat_rgbd(self):
        rgb_best, depth_best = self.get_best_modality_layers()
        self.train_labels, self.test_labels = [], []
        self.params.data_type = DataTypes.RGB
        rgb_training_set = WashingtonDataset(self.params, phase='train', loader=cnn_or_rnn_features_loader)
        rgb_train_loader = torch.utils.data.DataLoader(rgb_training_set, self.params.batch_size, shuffle=False)

        rgb_test_set = WashingtonDataset(self.params, phase='test', loader=cnn_or_rnn_features_loader)
        rgb_test_loader = torch.utils.data.DataLoader(rgb_test_set, self.params.batch_size, shuffle=False)

        rgb_rnn_train_out = []
        rgb_rnn_test_out = []
        for phase_loader in [rgb_train_loader, rgb_test_loader]:
            for inputs, labels, filenames in phase_loader:
                rgb_rnn_layer_feat = inputs[rgb_best].numpy()

                if phase_loader is rgb_train_loader:
                    rgb_rnn_train_out.append(rgb_rnn_layer_feat)
                else:
                    rgb_rnn_test_out.append(rgb_rnn_layer_feat)

                curr_labels = labels.numpy()
                if phase_loader == rgb_train_loader:
                    self.train_labels.append(curr_labels)
                else:
                    self.test_labels.append(curr_labels)

        rgb_rnn_train_out = np.concatenate([np.array(i) for i in rgb_rnn_train_out])
        rgb_rnn_test_out = np.concatenate([np.array(i) for i in rgb_rnn_test_out])

        self.params.data_type = DataTypes.Depth
        depth_training_set = WashingtonDataset(self.params, phase='train', loader=cnn_or_rnn_features_loader)
        depth_train_loader = torch.utils.data.DataLoader(depth_training_set, self.params.batch_size, shuffle=False)

        depth_test_set = WashingtonDataset(self.params, phase='test', loader=cnn_or_rnn_features_loader)
        depth_test_loader = torch.utils.data.DataLoader(depth_test_set, self.params.batch_size, shuffle=False)

        depth_rnn_train_out = []
        depth_rnn_test_out = []
        for phase_loader in [depth_train_loader, depth_test_loader]:
            for inputs, labels, filenames in phase_loader:
                depth_rnn_layer_feat = inputs[depth_best].numpy()

                if phase_loader is depth_train_loader:
                    depth_rnn_train_out.append(depth_rnn_layer_feat)
                else:
                    depth_rnn_test_out.append(depth_rnn_layer_feat)

        depth_rnn_train_out = np.concatenate([np.array(i) for i in depth_rnn_train_out])
        depth_rnn_test_out = np.concatenate([np.array(i) for i in depth_rnn_test_out])
        self.convert_labels()

        logging.info('Running Layer-[RGB_{} + Depth_{}]...'.format(rgb_best, depth_best))
        logging.info('Concat results of RGB_{} + Depth_{}'.format(rgb_best, depth_best))
        self.classify_rnn_features(np.concatenate((rgb_rnn_train_out, depth_rnn_train_out), axis=1),
                                   np.concatenate((rgb_rnn_test_out, depth_rnn_test_out), axis=1))
        logging.info('----------\n')

    def combine_one__bests(self, rgb_best_score, depth_best_score, rgb_best, depth_best):
        logging.info('Running Layer-[RGB_{}+Depth_{}] Average of Confidences for RGBD...'.format(rgb_best, depth_best))
        logging.info('Average SVM confidence scores of RGB_{} and Depth_{} are averaged'.format(rgb_best, depth_best))
        logging.info('SVMs confidence average of rgb and depth')
        rgbd_avg_confidence = np.mean(np.array([rgb_best_score, depth_best_score]), axis=0)
        rgbd_avg_preds = np.argmax(rgbd_avg_confidence, axis=1)
        avg_res, true_preds, test_size = self.calc_scores(rgbd_avg_preds)
        logging.info('Combined Average conf result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))
        logging.info('----------\n')

        logging.info('Running Layer-[RGB_{}+Depth_{}] Weighted combined of Confidences for RGBD'
                     '...'.format(rgb_best, depth_best))
        logging.info('Weighted SVM confidence scores of RGB_{} and Depth_{} are combined'.format(rgb_best, depth_best))
        logging.info('SVMs confidence weighted combined of rgb and depth')

        w_rgb, w_depth = self.calc_modality_weights((rgb_best_score, depth_best_score))
        rgbd_avg_confidence = np.add(rgb_best_score * w_rgb[:, np.newaxis],
                                     depth_best_score * w_depth[:, np.newaxis])
        rgbd_avg_preds = np.argmax(rgbd_avg_confidence, axis=1)
        avg_res, true_preds, test_size = self.calc_scores(rgbd_avg_preds)
        logging.info('Combined Weighted conf result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))

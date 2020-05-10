import numpy as np

import base_model
import model_utils
from base_model import Model
from basic_utils import DataTypes, RunSteps


class AlexNet(Model):

    def model_structure(self):
        rnn_layer_inp = {
            'layer1': (64, 27, 27),  # <- 64 x 27 x 27
            'layer2': (64, 13, 13),  # <- 192 x 13 x 13
            'layer3': (64, 13, 13),  # <- 384 x 13 x 13
            'layer4': (64, 26, 26),  # <- 256 x 13 x 13
            'layer5': (64, 12, 12),  # <- 256 x 6 x 6
            'layer6': (64, 8, 8),  # <- 4096
            'layer7': (64, 8, 8)  # <- 4096
        }
        return rnn_layer_inp

    def model_reduction_plan(self):
        # num_split, chunk_size, rfs, opt
        model_reduction = {
            'layer1': [(1, 1, 1, None)],
            'layer2': [(3, 64, 13, 'reduce_map')],
            'layer3': [(6, 64, 13, 'reduce_map')],
            'layer4': [(1, 1, 1, None)],
            'layer5': [(1, 1, 1, None)],
            'layer6': [(1, 1, 1, None)],
            'layer7': [(1, 1, 1, None)]
        }
        return model_reduction

    def process_layer1(self, curr_inputs):
        curr_layer = 'layer1'
        # no need for further processing, use as is
        pro_inp = curr_inputs
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer2(self, curr_inputs):
        curr_layer = 'layer2'
        model_reduction = self.model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = base_model.reduce_inp(weights, curr_inputs, num_split=num_split,
                                        pool_method=self.params.pooling, opt=opt)
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer3(self, curr_inputs):
        curr_layer = 'layer3'
        model_reduction = self.model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = base_model.reduce_inp(weights, curr_inputs, num_split=num_split,
                                        pool_method=self.params.pooling, opt=opt)
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer4(self, curr_inputs):
        curr_layer = 'layer4'
        pro_inp = model_utils.reshape_4d(curr_inputs, shape=(64, 26, 26))
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer5(self, curr_inputs):
        curr_layer = 'layer5'
        pro_inp = model_utils.reshape_4d(curr_inputs, shape=(64, 12, 12))
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer6(self, curr_inputs):
        curr_layer = 'layer6'
        pro_inp = model_utils.reshape_4d(curr_inputs, shape=(64, 8, 8))
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer7(self, curr_inputs):
        curr_layer = 'layer7'
        pro_inp = model_utils.reshape_4d(curr_inputs, shape=(64, 8, 8))
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def get_best_trio_layers(self):
        if self.params.proceed_step == RunSteps.FIX_RECURSIVE_NN:
            if self.params.data_type == DataTypes.RGB:
                return "layer4", "layer5", "layer6"
            else:
                return "layer2", "layer3", "layer5"
        else:
            if self.params.data_type == DataTypes.RGB:
                return "layer3", "layer4", "layer5"
            else:
                return "layer5", "layer6", "layer7"

    def get_best_modality_layers(self):
        if self.params.proceed_step == RunSteps.FIX_RECURSIVE_NN:
            rgb_best, depth_best = 'layer4', 'layer3'
        else:
            rgb_best, depth_best = 'layer4', 'layer5'

        return rgb_best, depth_best

    def eval(self):
        if self.params.data_type == DataTypes.RGBD:
            self.combine_rgbd()
        else:
            super(AlexNet, self).eval()

    def __init__(self, params):
        super(AlexNet, self).__init__(params)



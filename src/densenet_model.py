import numpy as np

import base_model
import model_utils
from base_model import Model
from basic_utils import DataTypes, RunSteps


class DenseNet(Model):
    def model_structure(self):
        rnn_layer_inp = {
            'layer1': (64, 28, 28),  # <- 256 x 56 x 56
            'layer2': (64, 14, 14),  # <- 128 x 28 x 28
            'layer3': (64, 14, 14),  # <- 512 x 28 x 28
            'layer4': (64, 28, 28),  # <- 256 x 14 x 14
            'layer5': (64, 14, 14),  # <- 1024 x 14 x 14
            'layer6': (64, 14, 14),  # <- 512 x 7 x 7
            'layer7': (64, 14, 14)  # <- 1024 x 7 x 7
        }
        return rnn_layer_inp

    def model_reduction_plan(self):
        # num_split, chunk_size, rfs
        model_reduction = {
            'layer1': [(4, 64, 56, 'reduce_map'), (4, 64, 28, 'reduce_rfs')],
            'layer2': [(2, 64, 28, 'reduce_map'), (4, 64, 14, 'reduce_rfs')],
            'layer3': [(8, 64, 28, 'reduce_map'), (4, 64, 14, 'reduce_rfs')],
            'layer4': [(1, 1, 1, None)],
            'layer5': [(16, 64, 14, 'reduce_map')],
            'layer6': [(2, 256, 7, 'reduce_map')],
            'layer7': [(4, 256, 7, 'reduce_map')]
        }
        return model_reduction

    def process_layer1(self, curr_inputs):
        curr_layer = 'layer1'
        model_reduction = self.model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = base_model.reduce_inp(weights, curr_inputs, num_split=num_split,
                                        pool_method=self.params.pooling, opt=opt)

        num_split, _, _, opt = model_reduction[curr_layer][1]
        weights = self.reduction_weights[curr_layer][1]
        pro_inp = base_model.reduce_inp(weights, pro_inp, num_split=num_split,
                                        pool_method=self.params.pooling, opt=opt)

        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer2(self, curr_inputs):
        curr_layer = 'layer2'
        model_reduction = self.model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = base_model.reduce_inp(weights, curr_inputs, num_split=num_split,
                                        pool_method=self.params.pooling, opt=opt)

        num_split, _, _, opt = model_reduction[curr_layer][1]
        weights = self.reduction_weights[curr_layer][1]
        pro_inp = base_model.reduce_inp(weights, pro_inp, num_split=num_split,
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

        num_split, _, _, opt = model_reduction[curr_layer][1]
        weights = self.reduction_weights[curr_layer][1]
        pro_inp = base_model.reduce_inp(weights, pro_inp, num_split=num_split,
                                        pool_method=self.params.pooling, opt=opt)

        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer4(self, curr_inputs):
        curr_layer = 'layer4'
        pro_inp = model_utils.reshape_4d(curr_inputs, shape=(64, 28, 28))
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer5(self, curr_inputs):
        curr_layer = 'layer5'
        model_reduction = self.model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = base_model.reduce_inp(weights, curr_inputs, num_split=num_split,
                                        pool_method=self.params.pooling, opt=opt)
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer6(self, curr_inputs):
        curr_layer = 'layer6'
        model_reduction = self.model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = base_model.reduce_inp(weights, curr_inputs, num_split=num_split,
                                        pool_method=self.params.pooling, opt=opt)
        pro_inp = model_utils.reshape_4d(pro_inp, shape=(64, 14, 14))
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def process_layer7(self, curr_inputs):
        curr_layer = 'layer7'
        model_reduction = self.model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = base_model.reduce_inp(weights, curr_inputs, num_split=num_split,
                                        pool_method=self.params.pooling, opt=opt)
        pro_inp = model_utils.reshape_4d(pro_inp, shape=(64, 14, 14))
        assert np.shape(pro_inp)[1:4] == self.model_structure()[curr_layer]

        return pro_inp

    def get_best_trio_layers(self):
        if self.params.proceed_step == RunSteps.FIX_RECURSIVE_NN:
            return "layer5", "layer6", "layer7"
        else:
            return "layer5", "layer6", "layer7"

    def get_best_modality_layers(self):
        if self.params.proceed_step == RunSteps.FIX_RECURSIVE_NN:
            rgb_best, depth_best = 'layer6', 'layer6'
        else:
            rgb_best, depth_best = 'layer6', 'layer7'

        return rgb_best, depth_best

    def eval(self):
        if self.params.data_type == DataTypes.RGBD:
            self.combine_rgbd()
        else:
            super(DenseNet, self).eval()

    def __init__(self, params):
        super(DenseNet, self).__init__(params)

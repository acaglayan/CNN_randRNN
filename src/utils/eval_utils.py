import argparse
import fnmatch
import os
import shutil

import h5py as h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

import sunrgbd
import wrgbd51
from alexnet_model import AlexNet
from basic_utils import Models, RunSteps
from densenet_model import DenseNet
from main import init_save_dirs
from resnet_models import ResNet
from vgg16_model import VGG16Net


def get_rnn_model(params):
    if params.net_model == Models.AlexNet:
        model_rnn = AlexNet(params)
    elif params.net_model == Models.VGGNet16:
        model_rnn = VGG16Net(params)
    elif params.net_model == Models.ResNet50 or params.net_model == Models.ResNet101:
        model_rnn = ResNet(params)
    else:  # params.net_model == Models.DenseNet121:
        model_rnn = DenseNet(params)

    return model_rnn


def calc_scores(l123_preds, test_labels, model_rnn):
    model_rnn.test_labels = test_labels
    avg_res, true_preds, test_size = model_rnn.calc_scores(l123_preds)
    conf_mat = confusion_matrix(test_labels, l123_preds)
    return avg_res, true_preds, test_size, conf_mat


def show_sunrgbd_conf_mat(conf_mat):
    num_ctgs = len(conf_mat)
    cm_sum = np.sum(conf_mat, axis=1, keepdims=True)
    cm_perc = conf_mat / cm_sum.astype(float) * 100
    columns = sunrgbd.get_class_names(range(num_ctgs))
    df_cm = pd.DataFrame(cm_perc, index=columns, columns=columns)
    plt.figure(figsize=(20, 15))
    sn.set(font_scale=1.4)  # for label size
    heatmap = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Oranges', fmt=".1f", vmax=100)  # font size
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=16)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=16)
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    plt.show()
    # plt.savefig('sunrgb_confusion_matrix.eps', format='eps', dpi=1000)


def calc_scores_conf_mat(svm_path):
    model_rnn = get_rnn_model(params)
    l1, l2, l3 = 'layer5', 'layer6', 'layer7'

    with h5py.File(svm_path, 'r') as f:
        l1_conf_scores = np.asarray(f[l1])
        l2_conf_scores = np.asarray(f[l2])
        l3_conf_scores = np.asarray(f[l3])
        test_labels = np.asarray(f['labels'])
    f.close()

    print('Running Layer-[{}+{}+{}] Confidence Average Fusion...'.format(l1, l2, l3))
    print('SVM confidence scores of {}, {} and {} are average fused'.format(l1, l2, l3))
    print('SVM confidence average fusion')
    l123_avr_confidence = np.mean(np.array([l1_conf_scores, l2_conf_scores, l3_conf_scores]), axis=0)
    l123_preds = np.argmax(l123_avr_confidence, axis=1)
    avg_res, true_preds, test_size, conf_mat = calc_scores(l123_preds, test_labels, model_rnn)
    print('Fusion result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))
    show_sunrgbd_conf_mat(conf_mat)


def sunrgbd_combined_scores_conf_mat(rgb_svm_path, depth_svm_path):
    model_rnn = get_rnn_model(params)
    l1, l2, l3 = 'layer5', 'layer6', 'layer7'
    with h5py.File(rgb_svm_path, 'r') as f:
        rgb1_conf_scores = np.asarray(f[l1])
        rgb2_conf_scores = np.asarray(f[l2])
        rgb3_conf_scores = np.asarray(f[l3])
        test_labels = np.asarray(f['labels'])
    f.close()

    with h5py.File(depth_svm_path, 'r') as f:
        depth1_conf_scores = np.asarray(f[l1])
        depth2_conf_scores = np.asarray(f[l2])
        depth3_conf_scores = np.asarray(f[l3])

    f.close()
    rgb_l123_sum_confidence = np.sum(np.array([rgb1_conf_scores, rgb2_conf_scores, rgb3_conf_scores]), axis=0)
    depth_l123_sum_confidence = np.sum(np.array([depth1_conf_scores, depth2_conf_scores, depth3_conf_scores]), axis=0)
    print('Weighted Average SVM confidence scores of [RGB({}+{}+{})+Depth({}+{}+{})] are taken')
    print('SVMs confidence weighted fusion')
    w_rgb, w_depth = model_rnn.calc_modality_weights((rgb_l123_sum_confidence, depth_l123_sum_confidence))
    rgbd_l123_wadd_confidence = np.add(rgb_l123_sum_confidence * w_rgb[:, np.newaxis],
                                       depth_l123_sum_confidence * w_depth[:, np.newaxis])
    l123_preds = np.argmax(rgbd_l123_wadd_confidence, axis=1)
    avg_res, true_preds, test_size, conf_mat = calc_scores(l123_preds, test_labels, model_rnn)
    print('Combined Weighted Confidence result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))
    show_sunrgbd_conf_mat(conf_mat)


def sunrgbd_main(params):
    root_path = '../../data/sunrgbd/'
    svm_conf_paths = root_path + params.features_root + params.proceed_step + '/svm_confidence_scores/'
    rgb_svm_path = svm_conf_paths + params.net_model + '_RGB_JPG.hdf5'
    depth_svm_path = svm_conf_paths + params.net_model + '_Depth_Colorized_HDF5.hdf5'

    if params.data_type == 'rgb':
        calc_scores_conf_mat(rgb_svm_path)
    elif params.data_type == 'depth':
        calc_scores_conf_mat(depth_svm_path)
    else:
        sunrgbd_combined_scores_conf_mat(rgb_svm_path, depth_svm_path)


def individual_class_scores(total_conf_mat):
    num_ctgs = len(total_conf_mat)
    cm_sum = np.sum(total_conf_mat, axis=1, keepdims=True)
    cm_perc = total_conf_mat / cm_sum.astype(float) * 100
    indidual_scores = cm_perc.diagonal()
    categories = wrgbd51.get_class_names(range(num_ctgs))
    i = 0
    for category, category_score in zip(categories, indidual_scores):
        print(f'{category:<15} {category_score:>10.1f}')


def show_wrgbd_conf_mat(conf_mat):
    num_ctgs = len(conf_mat)
    cm_sum = np.sum(conf_mat, axis=1, keepdims=True)
    cm_perc = conf_mat / cm_sum.astype(float) * 100
    columns = wrgbd51.get_class_names(range(num_ctgs))
    df_cm = pd.DataFrame(cm_perc, index=columns, columns=columns)
    plt.figure(figsize=(20, 15))
    sn.set(font_scale=1.4)  # for label size
    heatmap = sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Oranges', fmt=".1f", vmax=100)  # font size
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def wrgb_scores_conf_mat(params, svm_conf_paths):
    model_rnn = get_rnn_model(params)
    if params.data_type == 'rgb':
        params.proceed_step = RunSteps.FIX_RECURSIVE_NN
        data_type_ex = 'crop'
        params.data_type = 'crop'
        l1, l2, l3 = model_rnn.get_best_trio_layers()
        params.data_type = 'rgb'
    else:
        params.proceed_step = RunSteps.FINE_RECURSIVE_NN
        data_type_ex = 'depthcrop'
        params.data_type = 'depthcrop'
        l1, l2, l3 = model_rnn.get_best_trio_layers()
        params.data_type = 'depths'

    all_splits_scores = []
    for split in range(1, 11):
        conf_file = params.net_model + '_' + data_type_ex + '_split_' + str(split) + '.hdf5'
        svm_conf_file_path = svm_conf_paths + conf_file
        with h5py.File(svm_conf_file_path, 'r') as f:
            l1_conf_scores = np.asarray(f[l1])
            l2_conf_scores = np.asarray(f[l2])
            l3_conf_scores = np.asarray(f[l3])
            test_labels = np.asarray(f['labels'])
        f.close()
        # print('Running Layer-[{}+{}+{}] Confidence Average Fusion...'.format(l1, l2, l3))
        # print('SVM confidence scores of {}, {} and {} are average fused'.format(l1, l2, l3))
        # print('SVM confidence average fusion')
        l123_avr_confidence = np.mean(np.array([l1_conf_scores, l2_conf_scores, l3_conf_scores]), axis=0)
        l123_preds = np.argmax(l123_avr_confidence, axis=1)
        avg_res, true_preds, test_size, conf_mat = calc_scores(l123_preds, test_labels, model_rnn)
        # print('Fusion result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))
        all_splits_scores.append((avg_res, true_preds, test_size, conf_mat))

    total_avg_res = 0.0
    total_true_preds = 0.0
    total_test_size = 0.0
    total_conf_mat = np.zeros(shape=(51, 51), dtype=float)
    for avg_res, true_preds, test_size, conf_mat in all_splits_scores:
        total_avg_res += avg_res
        total_true_preds += true_preds
        total_test_size += test_size
        total_conf_mat += conf_mat

    print('Average score is {0:.1f}% ({1}/{2})'.format(total_avg_res / 10, total_true_preds, total_test_size))
    individual_class_scores(total_conf_mat)
    # show_wrgbd_conf_mat(total_conf_mat)


def wrgbd_combined_scores_conf_mat(params, svm_conf_paths):
    model_rnn = get_rnn_model(params)
    params.proceed_step = RunSteps.FIX_RECURSIVE_NN
    rgb_data_type_ex = 'crop'
    params.data_type = 'crop'
    rgb_l1, rgb_l2, rgb_l3 = model_rnn.get_best_trio_layers()

    params.proceed_step = RunSteps.FINE_RECURSIVE_NN
    depth_data_type_ex = 'depthcrop'
    params.data_type = 'depthcrop'
    depth_l1, depth_l2, depth_l3 = model_rnn.get_best_trio_layers()

    params.data_type = 'rgbd'
    all_splits_scores = []
    for split in range(1, 11):
        rgb_conf_file = params.net_model + '_' + rgb_data_type_ex + '_split_' + str(split) + '.hdf5'
        rgb_svm_conf_file_path = svm_conf_paths + rgb_conf_file
        with h5py.File(rgb_svm_conf_file_path, 'r') as f:
            rgb1_conf_scores = np.asarray(f[rgb_l1])
            rgb2_conf_scores = np.asarray(f[rgb_l2])
            rgb3_conf_scores = np.asarray(f[rgb_l3])
            test_labels = np.asarray(f['labels'])
        f.close()

        depth_conf_file = params.net_model + '_' + depth_data_type_ex + '_split_' + str(split) + '.hdf5'
        depth_svm_conf_file_path = svm_conf_paths + depth_conf_file
        with h5py.File(depth_svm_conf_file_path, 'r') as f:
            depth1_conf_scores = np.asarray(f[depth_l1])
            depth2_conf_scores = np.asarray(f[depth_l2])
            depth3_conf_scores = np.asarray(f[depth_l3])
        f.close()

        rgb_l123_sum_confidence = np.sum(np.array([rgb1_conf_scores, rgb2_conf_scores, rgb3_conf_scores]), axis=0)
        depth_l123_sum_confidence = np.sum(np.array([depth1_conf_scores, depth2_conf_scores, depth3_conf_scores]),
                                           axis=0)
        # print('Weighted Average SVM confidence scores of [RGB({}+{}+{})+Depth({}+{}+{})] are taken')
        # print('SVMs confidence weighted fusion')
        w_rgb, w_depth = model_rnn.calc_modality_weights((rgb_l123_sum_confidence, depth_l123_sum_confidence))
        rgbd_l123_wadd_confidence = np.add(rgb_l123_sum_confidence * w_rgb[:, np.newaxis],
                                           depth_l123_sum_confidence * w_depth[:, np.newaxis])
        l123_preds = np.argmax(rgbd_l123_wadd_confidence, axis=1)
        avg_res, true_preds, test_size, conf_mat = calc_scores(l123_preds, test_labels, model_rnn)
        # print('Combined Weighted Confidence result: {0:.2f}% ({1}/{2})..'.format(avg_res, true_preds, test_size))
        all_splits_scores.append((avg_res, true_preds, test_size, conf_mat))

    total_avg_res = 0.0
    total_true_preds = 0.0
    total_test_size = 0.0
    total_conf_mat = np.zeros(shape=(51, 51), dtype=float)
    for avg_res, true_preds, test_size, conf_mat in all_splits_scores:
        total_avg_res += avg_res
        total_true_preds += true_preds
        total_test_size += test_size
        total_conf_mat += conf_mat

    print('Average score is {0:.1f}% ({1}/{2})'.format(total_avg_res / 10, total_true_preds, total_test_size))
    individual_class_scores(total_conf_mat)


def wrgbd_main(params):
    root_path = '../../data/wrgbd/'
    svm_conf_paths = root_path + params.features_root + params.proceed_step + '/svm_confidence_scores/'
    if params.data_type == "rgbd":
        wrgbd_combined_scores_conf_mat(params, svm_conf_paths)
    else:
        wrgb_scores_conf_mat(params, svm_conf_paths)


def organize_dirs():
    dataset_dir = '../sunrgb_scene_dataset/RGB_JPG'
    for dir in os.listdir(dataset_dir):
        split_dir = os.path.join(dataset_dir, dir)
        for category in sorted(sunrgbd.class_names):
            suffix = category + '__*'
            save_dir = os.path.join(split_dir, category)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for file in fnmatch.filter(sorted(os.listdir(split_dir)), suffix):
                new_file_path = os.path.join(save_dir, file)
                current_file_path = os.path.join(split_dir, file)
                shutil.move(current_file_path, new_file_path)


def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", dest="task", default="scene", choices=["object", "scene"], type=str)
    parser.add_argument("--features-root", dest="features_root", default="outputs",
                        help="Root folder for CNN features to load/save")
    parser.add_argument("--net-model", dest="net_model", default=Models.ResNet101, choices=Models.ALL,
                        type=str.lower, help="Pre-trained network model to be employed as the feature extractor")
    parser.add_argument("--log-dir", dest="log_dir", default="../logs", help="Log directory")
    parser.add_argument("--data-type", dest="data_type", default="rgbd", choices=["rgb", "depth", "rgbd"])
    parser.add_argument("--debug-mode", dest="debug_mode", default=0, type=int, choices=[0, 1])

    params = parser.parse_args()
    params.proceed_step = RunSteps.OVERALL_RUN
    params.load_features = 1
    return params


if __name__ == '__main__':
    params = get_params()
    params = init_save_dirs(params)

    if params.task == "object":
        wrgbd_main(params)
    else:
        sunrgbd_main(params)

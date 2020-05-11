import argparse
import os
import time

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
import cv2 as cv
import matplotlib.pyplot as plt
import depth_transform
import recursive_nn
import sunrgbd
from basic_utils import RunSteps, PrForm, Pools
from extraction_models import ResNetExtractor
from loader_utils import sunrgbd_loader
from resnet_models import ResNet


class ResNetScene(ResNet):
    def get_best_trio_layers(self):
        return "layer5", "layer6", "layer7"

    def get_best_modality_layers(self):
        rgb_best, depth_best = 'layer6', 'layer7'

        return rgb_best, depth_best


def get_data_transform(data_type, std, mean):
    if data_type == "RGB_JPG":
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


def get_demo_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-path", dest="dataset_path", default="../../data/sunrgbd/",
                        help="Path to the data root")
    parser.add_argument("--features-root", dest="features_root", default="models-features/",
                        help="Root folder for CNN features to load/save")
    parser.add_argument("--data-type", dest="data_type", default="RGB_JPG", choices=["RGB_JPG"],
                        type=str, help="Data type to process, RGB has been given as an only sample demo type")
    parser.add_argument("--net-model", dest="net_model", default="resnet101", choices=["resnet101"],
                        type=str.lower, help="Pre-trained network model to be employed as the feature extractor")
    parser.add_argument("--num-rnn", dest="num_rnn", default=128, type=int, help="Number of RNN")
    parser.add_argument("--pooling", dest="pooling", default=Pools.RANDOM, choices=Pools.ALL,
                        type=str.lower, help="Pooling type")
    parser.add_argument("--reuse-randoms", dest="reuse_randoms", default=1, choices=[0, 1], type=int,
                        help="Handles whether the random weights are gonna save/load or not")
    parser.add_argument("--mode", dest="mode", default="image", choices=["camera", "image"], type=str,
                        help="Handles whether camera images or sample images are taken as inputs")
    parser.add_argument("--load-features", dest="load_features", default=0, type=int, choices=[0])
    # construct the argument parse and parse the arguments
    parser.add_argument("-o", "--output", type=str, default="demo_video",  # required=True,
                        help="path to output video file")
    parser.add_argument("-f", "--fps", type=int, default=10, help="FPS of output video")
    parser.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
    params = parser.parse_args()
    params.proceed_step = RunSteps.OVERALL_RUN
    return params


def take_sample_inputs(params, data_form, device, svm_estimators, model_ft, model_rnn, num_classes, test_img_dir):
    l5_svm_estimator, l6_svm_estimator, l7_svm_estimator = svm_estimators
    for file in sorted(os.listdir(test_img_dir)):
        path = os.path.join(test_img_dir, file)
        input_img = sunrgbd_loader(path, params)

        plt.imshow(np.asarray(input_img))

        input_img = data_form(input_img).unsqueeze(0)
        label = file[:file.find('__')]
        # label_id = np.int(sunrgbd.class_name_to_id[label])

        input_img = input_img.to(device)

        confidence_scores = []
        for extracted_layer, svm_model in zip(range(5, 8), (l5_svm_estimator, l6_svm_estimator, l7_svm_estimator)):
            extractor = ResNetExtractor(model_ft, extracted_layer, params.net_model)
            cnn_features = extractor(input_img).detach().cpu().clone().numpy()

            params.proceed_step = RunSteps.FINE_RECURSIVE_NN
            layer = 'layer' + str(extracted_layer)
            curr_layer_inp = model_rnn.process_layer(layer, cnn_features)
            curr_rnn_out = recursive_nn.forward_rnn(model_rnn.rnn_weights[layer], curr_layer_inp,
                                                    model_rnn.params.num_rnn, model_rnn.model_structure()[layer])
            params.proceed_step = RunSteps.OVERALL_RUN
            conf_score = svm_model.decision_function(curr_rnn_out)
            confidence_scores.append(conf_score)

        l123_avr_confidence = np.mean(np.array([confidence_scores[0], confidence_scores[1], confidence_scores[2]]),
                                      axis=0).squeeze()
        l123_preds_sorted = np.argsort(l123_avr_confidence)
        largest_conf_classes = l123_preds_sorted[::-1][:num_classes]

        top_1, top_2, top_3, top_4, top_5 = tuple(sunrgbd.get_class_names(largest_conf_classes[:5]))

        gt_str = 'ground-truth: '.rjust(15, ' ') + label.ljust(15, ' ')
        top_1_str = 'top-1: '.rjust(15, ' ') + top_1.ljust(15, ' ')
        top_2_str = 'top-2: '.rjust(15, ' ') + top_2.ljust(15, ' ')
        top_3_str = 'top-3: '.rjust(15, ' ') + top_3.ljust(15, ' ')
        top_4_str = 'top-4: '.rjust(15, ' ') + top_4.ljust(15, ' ')
        top_5_str = 'top-5: '.rjust(15, ' ') + top_5.ljust(15, ' ')

        plt.title('{}\n{}\n{}\n{}\n{}\n{}'.
                  format(gt_str, top_1_str, top_2_str, top_3_str,
                         top_4_str, top_5_str),  ha='center')
        plt.show()


def take_cam_inputs(params, data_form, device, svm_estimators, model_ft, model_rnn, num_classes, test_img_dir):
    l5_svm_estimator, l6_svm_estimator, l7_svm_estimator = svm_estimators
    video_out = test_img_dir + params.output
    # initialize the FourCC, video writer, dimensions of the frame, and
    # zeros array
    print(params.codec)
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    writer = None
    (h, w) = (None, None)
    zeros = None

    camera_ind = 0  # you can change this to 1, 2, 3 for other camera connections, 0 is default camera
    cap = cv.VideoCapture(camera_ind)
    while cap.isOpened():

        start = time.time()
        ret, input_org = cap.read()
        frame = input_org.copy()
        # input_img = custom_loader(path, params)

        # plt.imshow(np.asarray(input_img))

        input_img = Image.fromarray(input_org)
        input_img = data_form(input_img).unsqueeze(0)
        # label = file[:file.find('__')]
        # label_id = np.int(sunrgbd.class_name_to_id[label])

        input_img = input_img.to(device)

        confidence_scores = []
        for extracted_layer, svm_model in zip(range(5, 8), (l5_svm_estimator, l6_svm_estimator, l7_svm_estimator)):
            extractor = ResNetExtractor(model_ft, extracted_layer, params.net_model)
            cnn_features = extractor(input_img).detach().cpu().clone().numpy()

            params.proceed_step = RunSteps.FINE_RECURSIVE_NN
            layer = 'layer' + str(extracted_layer)
            curr_layer_inp = model_rnn.process_layer(layer, cnn_features)
            curr_rnn_out = recursive_nn.forward_rnn(model_rnn.rnn_weights[layer], curr_layer_inp,
                                                    model_rnn.params.num_rnn, model_rnn.model_structure()[layer])
            params.proceed_step = RunSteps.OVERALL_RUN
            conf_score = svm_model.decision_function(curr_rnn_out)
            confidence_scores.append(conf_score)

        l123_avr_confidence = np.mean(np.array([confidence_scores[0], confidence_scores[1], confidence_scores[2]]),
                                      axis=0).squeeze()
        l123_preds_sorted = np.argsort(l123_avr_confidence)
        largest_conf_classes = l123_preds_sorted[::-1][:num_classes]

        top_1, top_2, top_3, top_4, top_5 = tuple(sunrgbd.get_class_names(largest_conf_classes[:5]))
        # font
        font = cv.FONT_HERSHEY_SIMPLEX

        # org
        # org = (10, 10)
        # fontScale
        fontScale = 0.75

        # Blue color in BGR
        color = (0, 0, 0)

        # Line thickness of 2 px
        thickness = 1

        text1 = '\n top-1: {}\n top-2: {}\n top-3: {}\n top-4: {}\n top-5: {}'.format(top_1, top_2, top_3, top_4, top_5)
        y0, dy = 5, 25
        for i, line in enumerate(text1.split('\n')):
            y = y0 + i * dy
            cv.putText(input_org, line, (5, y), font, fontScale, color, thickness, cv.LINE_AA)
        # input_org = cv.putText(input_org, text1, org, font,
        #           fontScale, color, thickness, cv.LINE_AA)
        cv.imshow('Prediction', np.array(input_org))
        # check if the writer is None
        if writer is None:
            # store the image dimensions, initialize the video writer,
            # and construct the zeros array
            (h, w) = frame.shape[:2]
            writer = cv.VideoWriter(video_out, fourcc, params.fps, (w, h), True)
            zeros = np.zeros((h, w), dtype="uint8")
        # break the image into its RGB components, then construct the
        # RGB representation of each frame individually
        (B, G, R) = cv.split(frame)
        R = cv.merge([zeros, zeros, R])
        G = cv.merge([zeros, G, zeros])
        B = cv.merge([B, zeros, zeros])
        # construct the final output frame, storing the original frame
        # at the top-left, the red channel in the top-right, the green
        # channel in the bottom-right, and the blue channel in the
        # bottom-left
        output = np.zeros((h, w, 3), dtype="uint8")
        output[0:h, 0:w] = input_org

        # write the output frame to file
        writer.write(output)
        end = time.time() - start
        print(end)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # do a bit of cleanup
    print("[INFO] cleaning up...")
    cap.release()
    cv.destroyAllWindows()
    writer.release()


def run_demo(params):
    device = torch.device("cuda")
    model_rnn = ResNetScene(params)

    save_load_dir = params.dataset_path + params.features_root + params.proceed_step + '/svm_estimators/'
    svm_estimators_file = save_load_dir + params.net_model + '_' + params.data_type

    l5_svm_estimator_file = svm_estimators_file + '_l5.sav'
    l6_svm_estimator_file = svm_estimators_file + '_l6.sav'
    l7_svm_estimator_file = svm_estimators_file + '_l7.sav'

    l5_svm_estimator = joblib.load(l5_svm_estimator_file)
    l6_svm_estimator = joblib.load(l6_svm_estimator_file)
    l7_svm_estimator = joblib.load(l7_svm_estimator_file)

    test_img_dir = params.dataset_path + params.features_root + params.proceed_step + '/demo_images/'

    save_dir = params.dataset_path + params.features_root + RunSteps.FINE_TUNING + '/'
    best_model_file = save_dir + params.net_model + '_' + params.data_type + '_best_checkpoint.pth'

    num_classes = len(sunrgbd.class_names)
    model_ft = models.resnet101()

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    try:
        checkpoint = torch.load(best_model_file, map_location=device)
        model_ft.load_state_dict(checkpoint)
    except Exception as e:
        print('{}{}Failed to load the best model: {}{}'.format(PrForm.BOLD, PrForm.RED, e, PrForm.END_FORMAT))
        return

    # Set model to evaluation mode (without this, results will be completely different)
    # Remember that you must call model.eval() to set dropout and batch normalization layers
    # to evaluation mode before running inference.
    model_ft = model_ft.eval()
    model_ft = model_ft.to(device)

    std = [0.23089603, 0.2393163, 0.23400005]
    mean = [0.15509185, 0.16330947, 0.1496393]
    data_form = get_data_transform(params.data_type, std, mean)
    svm_estimators = (l5_svm_estimator, l6_svm_estimator, l7_svm_estimator)

    if params.mode == "camera":
        take_cam_inputs(params, data_form, device, svm_estimators, model_ft, model_rnn, num_classes, test_img_dir)
    else:
        take_sample_inputs(params, data_form, device, svm_estimators, model_ft, model_rnn, num_classes, test_img_dir)


def scene_demo_main():
    params = get_demo_params()
    run_demo(params)


if __name__ == '__main__':
    scene_demo_main()

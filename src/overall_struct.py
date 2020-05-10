import logging

import torch
from torch import nn
from torchvision import models

import basic_utils
import recursive_nn
import wrgbd51
from alexnet_model import AlexNet
from basic_utils import Models, RunSteps, PrForm, OverallModes, DataTypes
from densenet_model import DenseNet
from extraction_models import AlexNetVGG16Extractor, ResNetExtractor, DenseNet121Extractor
from loader_utils import custom_loader
from model_utils import get_data_transform
from resnet_models import ResNet
from vgg16_model import VGG16Net
from wrgbd_loader import WashingtonDataset


def process_rnn_stage(params, model_rnn, features, labels, filenames, phase, batch_ind):
    if params.run_mode == OverallModes.FINETUNE_MODEL:
        params.proceed_step = RunSteps.FINE_RECURSIVE_NN
    else:
        params.proceed_step = RunSteps.FIX_RECURSIVE_NN

    for extracted_layer in range(1, 8):
        layer = 'layer' + str(extracted_layer)
        curr_layer_inp = model_rnn.process_layer(layer, features[extracted_layer - 1])
        curr_rnn_out = recursive_nn.forward_rnn(model_rnn.rnn_weights[layer], curr_layer_inp,
                                                model_rnn.params.num_rnn, model_rnn.model_structure()[layer])

        if phase == 'train':
            model_rnn.rnn_train_outs[layer].append(curr_rnn_out)
        else:
            model_rnn.rnn_test_outs[layer].append(curr_rnn_out)

    curr_labels = labels.numpy()
    if phase == 'train':
        model_rnn.train_labels.append(curr_labels)
    else:
        model_rnn.test_labels.append(curr_labels)

    if model_rnn.params.save_features:
        model_rnn.save_recursive_features(filenames, batch_ind, phase=phase)

    params.proceed_step = RunSteps.OVERALL_RUN


@basic_utils.profile
def process_classification_stage(model_rnn, run_mode):
    logging.info('----------\n')
    model_rnn.convert_variables()
    l1_conf_scores = model_rnn.eval_layer1()
    logging.info('----------\n')

    l2_conf_scores = model_rnn.eval_layer2()
    logging.info('----------\n')

    l3_conf_scores = model_rnn.eval_layer3()
    logging.info('----------\n')

    l4_conf_scores = model_rnn.eval_layer4()
    logging.info('----------\n')

    l5_conf_scores = model_rnn.eval_layer5()
    logging.info('----------\n')

    l6_conf_scores = model_rnn.eval_layer6()
    logging.info('----------\n')

    l7_conf_scores = model_rnn.eval_layer7()
    logging.info('----------\n')

    model_rnn.save_svm_conf_scores(l1_conf_scores, l2_conf_scores, l3_conf_scores, l4_conf_scores, l5_conf_scores,
                                   l6_conf_scores, l7_conf_scores)
    if run_mode == OverallModes.FIX_PRETRAIN_MODEL:
        model_rnn.params.proceed_step = RunSteps.FIX_RECURSIVE_NN
    else:
        model_rnn.params.proceed_step = RunSteps.FINE_RECURSIVE_NN

    model_rnn.fusion_layers()
    logging.info('----------\n')
    model_rnn.params.proceed_step = RunSteps.OVERALL_RUN


def process_fusion(model_rnn, params):
    if params.data_type == DataTypes.RGBD:
        model_rnn.combine_rgbd()
    else:
        model_rnn.fusion_layers()

    logging.info('----------\n')


@basic_utils.profile
def run_overall_steps(params):
    # "cuda" if torch.cuda.is_available() else "cpu" instead of that I force to use cuda here
    device = torch.device("cuda")
    logging.info('Using device "{}"'.format(device))

    if params.net_model == Models.AlexNet:
        model_rnn = AlexNet(params)
    elif params.net_model == Models.VGGNet16:
        model_rnn = VGG16Net(params)
    elif params.net_model == Models.ResNet50 or params.net_model == Models.ResNet101:
        model_rnn = ResNet(params)
    else:  # params.net_model == Models.DenseNet121:
        model_rnn = DenseNet(params)

    if params.run_mode == OverallModes.FUSION:
        process_fusion(model_rnn, params)

    else:
        if params.run_mode == OverallModes.FINETUNE_MODEL:
            save_dir = params.dataset_path + params.features_root + RunSteps.FINE_TUNING + '/'
            best_model_file = save_dir + params.net_model + '_' + params.data_type + '_split_' + \
                              str(params.split_no) + '_best_checkpoint.pth'

            num_classes = len(wrgbd51.class_names)
            if params.net_model == Models.DenseNet121:
                model_ft = models.densenet121()
                num_ftrs = model_ft.classifier.in_features
                model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            elif params.net_model in (Models.ResNet50, Models.ResNet101):
                if params.net_model == Models.ResNet50:
                    model_ft = models.resnet50()
                else:
                    model_ft = models.resnet101()

                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, num_classes)
            else:  # params.net_model == Models.AlexNet or Models.VGGNet16
                if params.net_model == Models.AlexNet:
                    model_ft = models.alexnet()
                else:
                    model_ft = models.vgg16_bn()

                num_ftrs = model_ft.classifier[6].in_features
                model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

            try:
                checkpoint = torch.load(best_model_file, map_location=device)
                model_ft.load_state_dict(checkpoint)
            except Exception as e:
                print('{}{}Failed to load the finetuned model: {}{}'.format(PrForm.BOLD, PrForm.RED, e,
                                                                            PrForm.END_FORMAT))
                return
        elif params.run_mode == OverallModes.FIX_PRETRAIN_MODEL:

            if params.net_model == Models.AlexNet:
                model_ft = models.alexnet(pretrained=True)
            elif params.net_model == Models.VGGNet16:
                model_ft = models.vgg16_bn(pretrained=True)
            elif params.net_model == Models.ResNet50:
                model_ft = models.resnet50(pretrained=True)
            elif params.net_model == Models.ResNet101:
                model_ft = models.resnet101(pretrained=True)
            else:  # params.net_model is Models.DenseNet121
                model_ft = models.densenet121(pretrained=True)

        # Set model to evaluation mode (without this, results will be completely different)
        # Remember that you must call model.eval() to set dropout and batch normalization layers
        # to evaluation mode before running inference.
        model_ft = model_ft.eval()
        model_ft = model_ft.to(device)

        data_form = get_data_transform(params.data_type)

        training_set = WashingtonDataset(params, phase='train', loader=custom_loader, transform=data_form)
        train_loader = torch.utils.data.DataLoader(training_set, params.batch_size, shuffle=False)

        test_set = WashingtonDataset(params, phase='test', loader=custom_loader, transform=data_form)
        test_loader = torch.utils.data.DataLoader(test_set, params.batch_size, shuffle=False)

        data_loaders = {'train': train_loader, 'test': test_loader}

        for phase in ['train', 'test']:
            batch_ind = 0
            for inputs, labels, filenames in data_loaders[phase]:
                inputs = inputs.to(device)

                features = []
                for extracted_layer in range(1, 8):
                    if params.net_model == Models.AlexNet or params.net_model == Models.VGGNet16:
                        extractor = AlexNetVGG16Extractor(model_ft, extracted_layer, params.net_model)
                    elif params.net_model == Models.ResNet50 or params.net_model == Models.ResNet101:
                        extractor = ResNetExtractor(model_ft, extracted_layer, params.net_model)
                    else:  # params.net_model == Models.DenseNet121:
                        extractor = DenseNet121Extractor(model_ft, extracted_layer)

                    features.append(extractor(inputs).detach().cpu().clone().numpy())

                process_rnn_stage(params, model_rnn, features, labels, filenames, phase, batch_ind)
                batch_ind += 1

        process_classification_stage(model_rnn, params.run_mode)

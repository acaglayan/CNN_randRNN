import logging
import os

import h5py
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

import wrgbd51
from extraction_models import AlexNetVGG16Extractor, DenseNet121Extractor, ResNetExtractor
from model_utils import get_data_transform
from basic_utils import profile, PrForm, RunSteps, Models
from loader_utils import custom_loader
from wrgbd_loader import WashingtonAll, WashingtonDataset


@profile
def fixed_extraction(params):
    # "cuda" if torch.cuda.is_available() else "cpu" instead of that force for cuda :)
    device = torch.device("cuda")
    logging.info('Using device "{}"'.format(device))

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
    model_ft = model_ft.eval()
    model_ft = model_ft.to(device)

    data_form = get_data_transform(params.data_type)

    wrgbd = WashingtonAll(params, loader=custom_loader, transform=data_form)
    data_loader = torch.utils.data.DataLoader(wrgbd, params.batch_size, shuffle=False)

    for inputs, outs in data_loader:
        inputs = inputs.to(device)

        features = []
        for extracted_layer in range(1, 8):
            if params.net_model == Models.AlexNet or params.net_model == Models.VGGNet16:
                extractor = AlexNetVGG16Extractor(model_ft, extracted_layer, params.net_model)
            elif params.net_model == Models.ResNet50 or params.net_model == Models.ResNet101:
                extractor = ResNetExtractor(model_ft, extracted_layer, params.net_model)
            else:  # params.net_model is Models.DenseNet121
                extractor = DenseNet121Extractor(model_ft, extracted_layer)

            features.append(extractor(inputs).detach().cpu().clone().numpy())

        for i in range(len(outs)):

            with h5py.File(outs[i], 'w') as f:
                for extracted_layer in range(1, 8):
                    feature_type = 'layer' + str(extracted_layer)
                    f.create_dataset(feature_type, data=features[extracted_layer - 1][i, ])
            f.close()


@profile
def finetuned_extraction(params):
    # "cuda" if torch.cuda.is_available() else "cpu" instead of that force for cuda
    device = torch.device("cuda")
    logging.info('Using device "{}"'.format(device))

    save_dir = params.dataset_path + params.features_root + RunSteps.FINE_TUNING + '/'
    best_model_file = save_dir + params.net_model + '_' + params.data_type + '_split_' + str(params.split_no) + \
                      '_best_checkpoint.pth'

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
        print('{}{}Failed to load the model: {}{}'.format(PrForm.BOLD, PrForm.RED, e, PrForm.END_FORMAT))
        return

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

    results_dir = save_dir + 'split-' + str(params.split_no) + '/' + params.net_model + '_results_' + params.data_type
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    data_loaders = {'train': train_loader, 'test': test_loader}

    for phase in ['train', 'test']:
        for inputs, labels, filenames in data_loaders[phase]:
            inputs = inputs.to(device)

            features = []
            for extracted_layer in range(1, 8):
                if params.net_model == Models.AlexNet or params.net_model == Models.VGGNet16:
                    extractor = AlexNetVGG16Extractor(model_ft, extracted_layer, params.net_model)
                elif params.net_model == Models.ResNet50 or params.net_model == Models.ResNet101:
                    extractor = ResNetExtractor(model_ft, extracted_layer, params.net_model)
                else:   # params.net_model == Models.DenseNet121:
                    extractor = DenseNet121Extractor(model_ft, extracted_layer)

                features.append(extractor(inputs).detach().cpu().clone().numpy())

            for i in range(len(filenames)):

                with h5py.File(os.path.join(results_dir, filenames[i] + '.hdf5'), 'w') as f:
                    for extracted_layer in range(1, 8):
                        feature_type = 'layer' + str(extracted_layer)
                        f.create_dataset(feature_type, data=features[extracted_layer - 1][i,])
                f.close()

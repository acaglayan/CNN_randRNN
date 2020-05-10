import copy
import logging
import os
import time

import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torchvision import models

from basic_utils import RunSteps, Models
from model_utils import get_data_transform
import wrgbd51
from loader_utils import custom_loader
from wrgbd_loader import WashingtonDataset


def train_model(model, data_loaders, criterion, optimizer, scheduler, device, num_epochs=25):
    logging.info('Using device "{}"'.format(device))
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # call model.eval() to set dropout and batch normalization layers
                # to evaluation mode before running inference.
                # Failing to do this will yield inconsistent inference results.
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info('-' * 10)
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history


def process_finetuning(params):
    num_classes = len(wrgbd51.class_names)
    # uncomment saving codes after param search
    save_dir = params.dataset_path + params.features_root + RunSteps.FINE_TUNING + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_model_file = save_dir + params.net_model + '_' + params.data_type + '_split_' + str(params.split_no) + \
                      '_best_checkpoint.pth'

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    if params.net_model == Models.DenseNet121:
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    elif params.net_model in (Models.ResNet50, Models.ResNet101):
        if params.net_model == Models.ResNet50:
            model_ft = models.resnet50(pretrained=True)
        else:
            model_ft = models.resnet101(pretrained=True)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    else:  # params.net_model == 'alexnet'
        if params.net_model == Models.AlexNet:
            model_ft = models.alexnet(pretrained=True)
        else:
            model_ft = models.vgg16_bn(pretrained=True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    # following two lines are added for the second stage of the two-stage fine-tuning
    # state_dict = torch.load(best_model_file)
    # model_ft.load_state_dict(state_dict)

    # Parameters of newly constructed modules have "requires_grad=True" by default
    set_parameters_requires_grad(model_ft, params.net_model, train_only_one_layer=False)

    data_form = get_data_transform(params.data_type)

    training_set = WashingtonDataset(params, phase='train', loader=custom_loader, transform=data_form)
    train_loader = torch.utils.data.DataLoader(training_set, params.batch_size, shuffle=True)

    val_set = WashingtonDataset(params, phase='test', loader=custom_loader, transform=data_form)
    val_loader = torch.utils.data.DataLoader(val_set, params.batch_size, shuffle=False)
    data_loaders = {'train': train_loader, 'val': val_loader}

    # first stage of finetuning: finetune the last layer, freeze the rest of the network
    model_ft = fine_tuning(params, model_ft, data_loaders, device, stage=1)

    # report_cpu_stats()
    # report_gpu_memory()
    torch.save(model_ft.state_dict(), best_model_file)  # uncomment this line after param search


def set_parameters_requires_grad(model, net_model, train_only_one_layer):
    if not train_only_one_layer:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

        if net_model == Models.AlexNet or net_model == Models.VGGNet16:
            classifier = model.classifier[6]
        elif net_model == Models.DenseNet121:
            classifier = model.classifier
        else:   # Models.ResNet50, Models.ResNet101
            classifier = model.fc

        for param in classifier.parameters():
            param.requires_grad = True


def fine_tuning(params, model, data_loaders, device, stage):
    logging.info('-' * 20)
    logging.info('Stage-{} of fine-tuning is started..'.format(stage))
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum)
    optimizer_ft.zero_grad()

    # Decay LR by a factor of gamma every step_size epochs by default
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=params.step_size, gamma=params.gamma)

    model_ft, _ = train_model(model, data_loaders, criterion, optimizer_ft, exp_lr_scheduler, device,
                              num_epochs=params.num_epochs)

    return model_ft

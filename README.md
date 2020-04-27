# When CNNs Meet Random RNNs: Towards Multi-Level Analysis for RGB-D Object and Scene Recognition
![Overview of the two-stage framework](https://github.com/acaglayan/CNN_randRNN/blob/master/figures/overview.png)
## Introduction
This repository presents the implementation of a general two-stage framework that employs a convolutional neural network (CNN) model as the underlying feature extractor and random recursive neural network (RNN) to encode these features to high-level representations. A random weighted pooling approach, which is applicable to both spatial and channel wise downsampling, has been proposed by extending the idea of randomness in RNNs. Thus, the second stage of the framework presents a fully randomized stucture to encode CNN features efficiently. The framework is a general PyTorch-based codebase for RGB-D object and scene recognition and applicable to a variety of pretrained CNN models including AlexNet, VGGNet (VGGNet-16 in particular), ResNet (with two different variations ResNet-50 and ResNet-101) and DenseNet (DenseNet-101 specifically). The overall structure has been designed in a modular and extendable way through a unified CNN and RNN process. Therefore, it offers an easy and flexible use. These also can easily be extended with new capabilities and combinations with different setups and other models for implementing new ideas.

### Feature Highlights
- Support both one-stage CNN feature extraction and two-stage incorporation of CNN-randRNN feature extraction.
- Applicable to AlexNet, VGGNet-16, ResNet-50, ResNet-101, DenseNet-121 as backbone CNN models.
- Pretrained models can be used as fixed feature extractors in a fast way. They also can be used after performing finetuning.
- A novel random pooling strategy based on uniform randomness in RNNs is presented to cope with the high dimensionality of inputs.
- A soft voting approach based on individual SVM confidences for multi-modal fusion has been presented.
- An effective depth colorization based on surface normals has been presented.
- Clear and extendible code structure for supporting more datasets and applying to new ideas.
## Model Zoo
Supported backbone models and their average computational time and memory overhead for overall data processing and model learning on Washington RGB-D Object dataset are shown in the below table. These are the overall results of both of train and test phases together. Experiments are performed on a desktop PC with AMD Ryzen 9 3900X 12-Core Processor, 3.8 GHz Base, 128 GB DDR4 RAM 2666 MHz, and NVIDIA GeForce GTX 1080 Ti graphics card with 11 GB memory. The batch size is 64 for all the models.

![](https://raw.githubusercontent.com/acaglayan/CNN_randRNN/master/figures/model_table.png?token=AFBFTESP5MJLQECJCHRPKX26U4O4E)


## Installation
### Requirements
### Setup 

## Getting Started

## Acknowledgment

## License

## Citation

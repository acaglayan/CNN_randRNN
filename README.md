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
## Installation
### Requirements
### Setup 

## Getting Started

## Acknowledgment

## License

## Citation

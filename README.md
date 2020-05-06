# When CNNs Meet Random RNNs: Towards Multi-Level Analysis for RGB-D Object and Scene Recognition
![Overview of the two-stage framework](https://github.com/acaglayan/CNN_randRNN/blob/master/figures/overview.png)
## Introduction
This repository presents the implementation of a general two-stage framework that employs a convolutional neural network (CNN) model as the underlying feature extractor and random recursive neural network (RNN) to encode these features to high-level representations. A random weighted pooling approach, which is applicable to both spatial and channel wise downsampling, has been proposed by extending the idea of randomness in RNNs. Thus, the second stage of the framework presents a fully randomized stucture to encode CNN features efficiently. The framework is a general PyTorch-based codebase for RGB-D object and scene recognition and applicable to a variety of pretrained CNN models including AlexNet, VGGNet (VGGNet-16 in particular), ResNet (with two different variations ResNet-50 and ResNet-101) and DenseNet (DenseNet-101 specifically).The overall structure has been designed in a modular and extendable way through a unified CNN and RNN process. Therefore, it offers an easy and flexible use. These also can easily be extended with new capabilities and combinations with different setups and other models for implementing new ideas.

This work has been tested on the popular <a href ="https://rgbd-dataset.cs.washington.edu/dataset.html" target="_blank">Washington RGB-D Object </a>  and <a href ="http://rgbd.cs.princeton.edu/" target="_blank">SUN RGB-D Scene</a> datasets demonstrating state-of-the-art results both in object and scene recognition tasks.  

### Feature Highlights
- Support both one-stage CNN feature extraction and two-stage incorporation of CNN-randRNN feature extraction.
- Applicable to AlexNet, VGGNet-16, ResNet-50, ResNet-101, DenseNet-121 as backbone CNN models.
- Pretrained models can be used as fixed feature extractors in a fast way. They also can be used after performing finetuning.
- A novel random pooling strategy, which extends the uniform randomness in RNNs, is presented to cope with the high dimensionality of inputs.
- A soft voting approach based on individual SVM confidences for multi-modal fusion has been presented.
- An effective depth colorization based on surface normals has been presented.
- Clear and extendible code structure for supporting more datasets and applying to new ideas.
## Model Zoo
Supported backbone models and their average computational time and memory overhead for overall data processing and model learning on Washington RGB-D Object dataset are shown in the below table. These are the overall results of both train and test phases together. Experiments are performed on a workstation with AMD Ryzen 9 3900X 12-Core Processor, 3.8 GHz Base, 128 GB DDR4 RAM 2666 MHz, and NVIDIA GeForce GTX 1080 Ti graphics card with 11 GB memory. The batch size is 64 for all the models.

![](https://github.com/acaglayan/CNN_randRNN/blob/master/figures/model_table.png)


## Installation
### System Requirements
All the codes are tested with the abovementioned environment. System requirements for each model are reported on the above model zoo table. Ideally, it would be better if you have a multi-core processor, 32 GB RAM, graphics card with at least 10 GB memory, and enough disk space to store models, features, etc. depending on your saving choices and initial parameters.
### Setup 
`conda` has been used as the virtual environment manager and `pip` as package manager. You can use either `pip` or `conda` (or both) for package management. Before starting you need to install following libraries:
* PyTorch
* Scikit-learn
* OpenCV
* psutil and h5py libs.
We have installed these libraries with `pip` as below:
1. Create virtual environment. <br/>
```
conda create -n cnnrandrnn python=3.7
source activate cnnrandrnn
```
1. Install Pytorch according to your system preferences such as OS, package manager, and CUDA version (see more [here](https://pytorch.org/get-started/locally/)): <br />
e.g. `pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html` <br />
This will install some other libs including `numpy`, `pillow`, etc. <br />

2. Install `scikit-learn`: <br />
e.g. `pip install -U scikit-learn` <br />

3. Install OpenCV library: <br />
e.g. `pip install opencv-python` <br />

4. Install `psutil` and `h5py` libs: <br />
e.g. `pip install psutil` <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`pip install h5py` <br />


## Getting Started
### Data Preparation
1- Washington RGB-D Object dataset is available <a href="https://rgbd-dataset.cs.washington.edu/dataset.html" target="_blank">here</a>. We have tested our framework using cropped evaluation set without extra background subtraction. Uncompress the data and place in `data/wrgbd` (see the structure below).
![](https://github.com/acaglayan/CNN_randRNN/blob/master/figures/wrgbd_view.png)

To convert depth maps to colorized RGB-like depth representations:
```
sh run_steps.sh step="COLORIZED_DEPTH_SAVE"
python main_steps.py --dataset-path "../data/wrgbd/" --data-type "depthcrop" --debug-mode 0
```
Note that you might need to export `/src/utils` to the PYTHONPATH (e.g. `export PYTHONPATH=$PYTHONPATH:/home/user/path_to_project/CNN_randRNN/src/utils`). `debug-mode` with 1 runs the framework for a small proportion of data (you can choose the size with `debug-size` parameter, which sets the number of samples for each instance.) This will create colorized depth images under the `/data/wrgbd/outputs/colorized_depth_images`.

2- 

## Acknowledgment
This  paper  is  based  on  the  results  obtained  from  a  project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
## License

## Citation
If you find this work useful in your research, please consider citing:
```
@article{Caglayan2020CNNrandRNN,
  title={When CNNs Meet Random RNNs: Towards Multi-Level Analysis for RGB-D Object and Scene Recognition},
  author={Ali Caglayan and Nevrez Imamoglu and Ahmet Burak Can and Ryosuke Nakamura},
  journal={arXiv preprint arXiv:2004.12349},
  year={2020}
}
```

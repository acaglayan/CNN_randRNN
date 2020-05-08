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
2. Install Pytorch according to your system preferences such as OS, package manager, and CUDA version (see more [here](https://pytorch.org/get-started/locally/)): <br />
e.g. `pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html` <br />
This will install some other libs including `numpy`, `pillow`, etc. <br />

3. Install `scikit-learn`: <br />
e.g. `pip install -U scikit-learn` <br />

4. Install OpenCV library: <br />
e.g. `pip install opencv-python` <br />

5. Install `psutil` and `h5py` libs: <br />
e.g. `pip install psutil` <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`pip install h5py` <br />


## Getting Started
### Data Preparation
1- Washington RGB-D Object dataset is available <a href="https://rgbd-dataset.cs.washington.edu/dataset.html" target="_blank">here</a>. We have tested our framework using cropped evaluation set without extra background subtraction. Uncompress the data and place in `data/wrgbd` (see the structure below).
```
CNN_randRNN
├── data
│   ├── wrgbd
│   │   │──eval-set
│   │   │   ├──apple
│   │   │   ├──ball
│   │   │   ├──...
│   │   │   ├──water_bottle
│   │   │──split.mat
├── src
├── logs
```

To convert depth maps to colorized RGB-like depth representations:
```
sh run_steps.sh step="COLORIZED_DEPTH_SAVE"
python main_steps.py --dataset-path "../data/wrgbd/" --data-type "depthcrop" --debug-mode 0
```
Note that you might need to export `/src/utils` to the PYTHONPATH (e.g. `export PYTHONPATH=$PYTHONPATH:/home/user/path_to_project/CNN_randRNN/src/utils`). `debug-mode` with 1 runs the framework for a small proportion of data (you can choose the size with `debug-size` parameter, which sets the number of samples for each instance.) This will create colorized depth images under the `/data/wrgbd/outputs/colorized_depth_images`.

2- 

### Params for Overall Run
Before demonstrating how to run the program, let's see the command line parameters with their default values for running the program.<br/>
```
--dataset-path "../data/wrgbd/" 
```
This is the root path of the dataset. <br/>

```
 --features-root "outputs" 
```
This is the root folder for saving/loading models, features, weights, etc.<br/>

```
--data-type "crop" 
```
Data type to process, `crop` for rgb, `depthcrop` for depth data. And `rgbd` for multi-modal fusion. <br/>

```
--net-model "alexnet" 
```
Backbone CNN model to be employed as the feature extractor. Could be one of these: `alexnet`, `vgg16_bn`, `resnet50`, `resnet101`, and `densenet121`. <br/>

```
--debug-mode 1 
```
This controls to run with all of the dataset (`0`) or with a small proportion of dataset (`1`). Default value is `1` to check if everything is fine with setups etc.<br/>

```
--debug-size 3 
```
This determines the proportion size for debug-mode. The default value of `3` states that for every instance of a category, 3 samples are going to be taken to process.<br/>

```
--log-dir "../logs" 
```
This is the root folder for saving log files.<br/>

```
--batch-size 64 
```
You can set the batch size with this parameter.<br/>

```
--split-no 1 
```
There are 10 splits in total for Washington RGB-D Object dataset. This indicates the running split. It should be `1` to `10`.<br/>

```
--run-mode 2 
```
There are 3 run modes. `1` is to use the finetuned backbone models, `2` is to use fixed pretrained CNN models, and `3` is for fusion run. Before running for fusion (`3`), you should run the framework for RGB and depth first with run-mode `1` or `2`.<br/>

```
--num-rnn 128 
```
You can set the number of random RNN with this parameter.<br/>

```
--save-features 0 
```
If you want to save features, you can set this parameter to `1`.<br/>

```
--reuse-randoms 1 
```
This decides whether the already saved random weights are going to be used. If there are not available saved weights, it will save the weights for later runs. Otherwise, if it is set to `0`, weights are not going to saved/load and the program generates new random weights in each run.<br/>

```
--pooling "random"  
```
Pooling method can be one of `max`, `avg`, and `random`.<br/>

```
--load-features 0  
```
If the features are already saved (with the `--save-fatures 1`), it is possible to load them without the need for run the whole pipeline again by setting this parameter to `1`.<br/>

There is one other parameter `--trial`. This is a control param for multiple runs. It could be used for multiple runs to evaluate different parameters in a controlled way. 

### Run Overall Pipeline
To run the overall pipeline with the defaul parameter values:<br/>
```
python main.py
```
This will train/test SVM for every 7 layers. You may want to make levels other than that of optimum ones to the comment line.

### Run Individual Steps
To run individual steps:<br/>
```
sh run_steps.sh step="FIX_EXTRACTION"
python main_steps.py
```
`step` parameter of the shell command is one of `COLORIZED_DEPTH_SAVE`, `FIX_EXTRACTION`, `FIX_RECURSIVE_NN`, `FINE_TUNING`, `FINE_EXTRACTION`, and `FINE_RECURSIVE_NN`.

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

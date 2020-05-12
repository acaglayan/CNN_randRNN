# When CNNs Meet Random RNNs: Towards Multi-Level Analysis for RGB-D Object and Scene Recognition
This repository presents the implementation of a general two-stage framework for RGB-D object and scene recognition taskt. The framework employs a convolutional neural network (CNN) model as the underlying feature extractor and random recursive neural network (RNN) to encode these features into high-level representations. For the details, please refer to:<br/>
<br/>
**When CNNs Meet Random RNNs: Towards Multi-Level Analysis for RGB-D Object and Scene Recognition**<br/>
<a href="https://github.com/acaglayan" target="_blank">Ali Caglayan</a>, <a href="https://scholar.google.com/citations?hl=en&user=VJgx61MAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Nevrez Imamoglu</a>, <a href="https://scholar.google.com/citations?hl=en&user=SbRj1qwAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Ahmet Burak Can</a>, <a href="https://www.airc.aist.go.jp/en/gsrt/" target="_blank">Ryosuke Nakamura</a>
<br/>
[<a href="https://arxiv.org/pdf/2004.12349" target="_blank">Paper</a>]

<br/>

![Overview of the two-stage framework](https://github.com/acaglayan/CNN_randRNN/blob/master/figures/overview.png)

## Introduction
The framework is a general PyTorch-based codebase for RGB-D object and scene recognition. The overall structure has been designed in a modular and extendable way through a unified CNN and RNN process. Therefore, it offers an easy and flexible use. These also can be extended with new capabilities and combinations with different setups and other models for implementing new ideas.

This work has been tested on the popular <a href ="https://rgbd-dataset.cs.washington.edu/dataset.html" target="_blank">Washington RGB-D Object </a>  and <a href ="http://rgbd.cs.princeton.edu/" target="_blank">SUN RGB-D Scene</a> datasets demonstrating state-of-the-art results both in object and scene recognition tasks.  

### Feature Highlights
- Support both one-stage CNN feature extraction and two-stage incorporation of CNN-randRNN feature extraction.
- Applicable to AlexNet, VGGNet-16, ResNet-50, ResNet-101, DenseNet-121 as backbone CNN models.
- Pretrained models can be used as fixed feature extractors in a fast way. They also can be used after performing finetuning.
- A novel random pooling strategy, which extends the uniform randomness in RNNs and is applicable to both spatial and channel wise downsampling, is presented to cope with the high dimensionality of CNN activations.
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
* psutil, h5py, seaborn, and matplotlib libs. <br/>
We have installed these libraries with `pip` as below:<br/>
1. Create virtual environment. <br/>
```
conda create -n cnnrandrnn python=3.7
conda activate cnnrandrnn
```
2. Install Pytorch according to your system preferences such as OS, package manager, and CUDA version (see more [here](https://pytorch.org/get-started/locally/)): <br />
e.g. `pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html` <br />
This will install some other libs including `numpy`, `pillow`, etc. <br />

3. Install `scikit-learn`: <br />
e.g. `pip install -U scikit-learn` <br />

4. Install OpenCV library: <br />
e.g. `pip install opencv-python` <br />

5. Install `psutil`, `h5py`, `seaborn` and `matplotlib` libs:
``` 
pip install psutil
pip install h5py
pip install seaborn
pip install -U matplotlib
```

## Getting Started
### File Structure
The following directory structure is a reference to run the code as described in this documentation. This structure can change according to the command line parameters.
<pre>
CNN_randRNN
├── data
│   ├── wrgbd
│   │   │──eval-set
│   │   │   ├──apple
│   │   │   ├──ball
│   │   │   ├──...
│   │   │   ├──water_bottle
│   │   │──split.mat
│   ├── sunrgbd
│   │   │──SUNRGBD
│   │   │   ├──kv1
│   │   │   ├──kv2
│   │   │   ├──realsense
│   │   │   ├──xtion
│   │   │──allsplit.mat
│   │   │──SUNRGBDMeta.mat
│   │   │──organized-set
│   │   │   ├──Depth_Colorized_HDF5
│   │   │   │  ├──test
│   │   │   │  ├──train
│   │   │   ├──RGB_JPG
│   │   │   │  ├──test
│   │   │   │  ├──train
│   │   │──models-features
│   │   │   ├──fine_tuning
│   │   │   │  ├──<b><i>resnet101_Depth_Colorized_HDF5_best_checkpoint.pth</i></b>
│   │   │   │  ├──<b><i>resnet101_RGB_JPG_best_checkpoint.pth</i></b>
│   │   │   ├──overall_pipeline_run
│   │   │   │  ├──svm_estimators
│   │   │   │  │  ├──<b><i>resnet101_Depth_Colorized_HDF5_l5.sav</i></b>
│   │   │   │  │  ├──<b><i>resnet101_Depth_Colorized_HDF5_l6.sav</i></b>
│   │   │   │  │  ├──<b><i>resnet101_Depth_Colorized_HDF5_l7.sav</i></b>
│   │   │   │  │  ├──<b><i>resnet101_RGB_JPG_l5.sav</i></b>
│   │   │   │  │  ├──<b><i>resnet101_RGB_JPG_l6.sav</i></b>
│   │   │   │  │  ├──<b><i>resnet101_RGB_JPG_l7.sav</i></b>
│   │   │   │  ├──demo_images
│   │   │   ├──random_weights
│   │   │   │  ├──<b><i>resnet101_reduction_random_weights.pkl</i></b>
│   │   │   │  ├──<b><i>resnet101_rnn_random_weights.pkl</i></b>
├── src
├── logs
</pre>

### Washington RGB-D Object Recognition
#### Data Preparation
Washington RGB-D Object dataset is available <a href="https://rgbd-dataset.cs.washington.edu/dataset.html" target="_blank">here</a>. We have tested our framework using cropped evaluation set without extra background subtraction. Uncompress the data and place in `data/wrgbd` (see the file structure above).

To convert depth maps to colorized RGB-like depth representations:
```
sh run_steps.sh step="COLORIZED_DEPTH_SAVE"
python main_steps.py --dataset-path "../data/wrgbd/" --data-type "depthcrop" --debug-mode 0
```
Note that you might need to export `/src/utils` to the PYTHONPATH (e.g. `export PYTHONPATH=$PYTHONPATH:/home/user/path_to_project/CNN_randRNN/src/utils`). `debug-mode` with 1 runs the framework for a small proportion of data (you can choose the size with `debug-size` parameter, which sets the number of samples for each instance.) This will create colorized depth images under the `/data/wrgbd/models-features/colorized_depth_images`.

#### Run Overall Pipeline
Before demonstrating how to run the program, see the explanations for command line parameters with their default values <a href="https://github.com/acaglayan/CNN_randRNN/blob/master/more_info.md"> here</a>. <br/>
To run the overall pipeline with the defaul parameter values:<br/>
```
python main.py
```
This will train/test SVM for every 7 layers. You may want to make levels other than that of optimum ones to the comment lines.
It is also possible to run the system step by step. See the details <a href="https://github.com/acaglayan/CNN_randRNN/blob/master/more_info.md"> here</a>.

### SUN RGB-D Scene Recognition
This codebase is presented based on Washington RGB-D object recognition. It can also be applied to SUN RGB-D Scene dataset. Please see the details <a href="https://github.com/acaglayan/CNN_randRNN/blob/master/sunrgbd_info.md"> here</a> to use SUN RGB-D Scene dataset. This can also be considered as a reference guide for the use of other datasets.

#### Scene Recognition Demo
A demo application using RGB images is presented. Download trained models and RNN random weights <a href="" target="__blank"> here <a/>. Uncompress the folder and place as the  file structure given above.
There are two run modes. To run the demo application with the default parameters for each mode is as below:<br/>
<table border="0">
  <tr>
   <td>
    <pre>python demo.py --mode "image"</pre>  
   </td>
   <td>
    <pre>python demo.py --mode "camera"</pre>
    </td>
 </tr>
   <tr>
   <td>
    <img align="center" src="https://github.com/acaglayan/CNN_randRNN/blob/master/figures/demo_samples.png">
   </td>
   <td>
    <img align="center" src="https://github.com/acaglayan/CNN_randRNN/blob/master/figures/demo_scene_recognition.gif">
    </td>
 </tr>
</table>
 <br/>  
  
`image` mode takes the images in the `demo_images` folder, while the `camera` modes takes camera images as inputs.

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

## License

## Acknowledgment
This  paper  is  based  on  the  results  obtained  from  a  project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).

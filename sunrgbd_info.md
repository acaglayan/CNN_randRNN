## SUN RGB-D Scene Recognition
### Data Preparation
SUN RGB-D Scene dataset is available <a href="http://rgbd.cs.princeton.edu/data/SUNRGBD.zip" target="_blank">here</a>. In addition, `allsplit.mat` and `SUNRGBDMeta.mat` files need to be downloaded from the SUN RGB-D toolbox.
In order to localize the paths provided in the `SUNRGBDMeta.mat` file and to make the dataset simple for the local system:
<br/>
```
sh run_steps.sh step="SAVE_SUNRGBD"
python main_steps.py --dataset-path "../data/sunrgbd/" --data-type "RGB_JPG" --debug-mode 0
```
This copies RGB images into `train/test` folders by renaming files with category information according to the provided `train/test` splits.
```
sh run_steps.sh step="SAVE_SUNRGBD"
python main_steps.py --dataset-path "../data/sunrgbd/" --data-type "Depth_Colorized_HDF5" --debug-mode 0
```
This converts depth maps to the proposed colorized RGB-like representations using the provided camera intrinsic values and saves files in `train/test` folders and `hdf5` file format. See the file structure <a href="https://github.com/acaglayan/CNN_randRNN/edit/master/README.md" target="_blank">here</a> for the saved files location.
<br/>

Note that, data preparation works quite slowly especially for the depth data. Nevertheless, it is needed to run just once. <br/>

### Preparing Source Codes

### Params for Overall Run
The command line parameters to run the overall pipeline:<br/>
```
--dataset-path "../data/sunrgbd/" 
```
This is the root path of the dataset. <br/>

```
 --features-root "models-features" 
```
This is the root folder for saving/loading models, features, weights, etc.<br/>

```
--data-type "RGB_JPG" 
```
Data type to process, `RGB_JPG` for rgb, `Depth_Colorized_HDF5` for depth data. And `RGBD` for multi-modal fusion. <br/>

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
This decides whether the already saved random weights are going to be used. If there are not available saved weights, it will save the weights for later runs. Otherwise, if it is set to `0`, weights are not going to saved/load and the program generates new random weights for each run.<br/>

```
--pooling "random"  
```
Pooling method can be one of `max`, `avg`, and `random`.<br/>

```
--load-features 0  
```
If the features are already saved (with the `--save-fatures 1`), it is possible to load them without the need for run the whole pipeline again by setting this parameter to `1`.<br/>

There is one other parameter `--trial`. This is a control param for multiple runs. It could be used for multiple runs to evaluate different parameters in a controlled way. 


#### Run Individual Steps
See <a href="https://github.com/acaglayan/CNN_randRNN/blob/master/more_info.md"> here</a> for the details to run individual steps.

<br/><br/>

<p align="center">
      <a href="https://github.com/acaglayan/CNN_randRNN">Back to Home Page</a>
 </p>

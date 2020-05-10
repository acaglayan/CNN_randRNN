### Params for Overall Run
The command line parameters with their default values for running the program:<br/>
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


#### Run Individual Steps
To run individual steps:<br/>
```
sh run_steps.sh step="FIX_EXTRACTION"
python main_steps.py
```
`step` parameter of the shell command is one of the below parameters: <br/>

- `COLORIZED_DEPTH_SAVE` converts depth maps to colorized RGB-like representations. <br/>
- `FIX_EXTRACTION` is for the use of fixed pretrained CNN models without training. <br/>
- `FIX_RECURSIVE_NN` runs multiple random RNNs using fixed pretrained CNN models. <br/>
- `FINE_TUNING` trains (finetunes) pretrained CNN models. <br/>
- `FINE_EXTRACTION` is for the use of finetuned CNN models as backbone models. <br/>
- `FINE_RECURSIVE_NN` runs multiple random RNNs using finetuned CNN models. <br/>
- `SAVE_SUNRGBD` this re-organizes SUN RGB-D Scene dataset by copying RGB and colorized depth images into train/test splits. Note that, this works quite slowly especially for depth data. Nevertheless, it is needed to run just once. <br/>

Check the source code for each individual step's command line parameters. See the paper for training hyperparameters.

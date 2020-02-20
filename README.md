# TL_blogpost
Transfer Learning with the OPU - Blogpost code 
Original blogpost is available at [this link](https://medium.com/@LightOnIO/au-revoir-backprop-bonjour-optical-transfer-learning-5f5ae18e4719)

## How to install

We advise creating a `virtualenv` before running these commands. You can create one with `python3 -m venv <venv_name>`. 
Activate it with source `<path_to_venv>/bin/activate`  before proceeding. We used `python 3.5` and `pytorch 1.2` 
for all the simulations.

- Clone the repository and then do `pip install <path_to_repo>`.

- (optional) Should you wish to replicate the results with `TensorRT` in `int8` you need to download the appropriate version from the
 [official NVIDIA website](https://developer.nvidia.com/tensorrt). We tested the code with `TensorRT 6.0.1.5` with `CUDA 10.1`.

- Finally download the dataset from the [Kaggle page](https://www.kaggle.com/alessiocorrado99/animals10). You should put the dataset 
in the  same folder as the repo, but all scripts have an option to change the path with `-dataset_path`. 

**NOTE**: we had problems with the `Pillow` package because this combination of Pytorch and TensorRT requires version
 `Pillow 6.1` in the `onnx` conversion of the model. If you have the same problems, uninstall `Pillow` and then retry with
 `pip install Pillow==6.1`. 

## Replicate the OPU/backprop results

Use the script `multiple_block.sh` in the `bash` folder. Open it in a text editor and then:
- set the OPU/backprop flags at the top to `true`, depending on which simulation you want to run
- Set the dtype to `float32`/`float16`. This affects only the OPU simulation.

- (optional) change the path to the script/dataset/save folder if you want to deviate from the 
defaults; 

- launch `./multiple_block.sh`. You might need to run `chmod +x multiple_block.sh` to make the script executable.

#### Jupyter notebook option
The notebook `TL_OPU.ipynb` in the `notebooks` folder does largely the same thing as the OPU script. It is a good way 
to get an idea of the general pipeline on the full DenseNet model.

## Replicate the TensorRT results  

Navigate to the `script` folder and then launch the following command: 

```
python3 tensorrt_training.py densenet169 Saturn -dtype_train int8 -dtype_inf int8 -block 10 -layer 12 
-n_components 2 -encode_type plain_th -encode_thr 0 -alpha_exp_min 6 -alpha_exp_max 8 
-save_path ~/dummy/int8/ -features_path ~/datasets_conv_features/int8_features/
``` 

Substitute the `save_path` with your desired destination folder. In the above example I had pre-extracted the features 
on a GPU which supported `int8` (RTX 2080) and then moved them to the OPU machine. If your machine already supports 
`int8` just drop the `-features_path` argument.

If you want to just extract the dataset features you can use the `tensorrt_extract_features.py`. Example call:

```
python3 tensorrt_extract_features.py densenet169 32 -block 10 -layer 12 
-dtype_train int8 -dtype_inf int8 -dataset_path ~/datasets/animals10/
```

Obviously change the dataset path with the correct one on your machine.

## Hardware specifics

All the simulations have been run on a Tesla P100 GPU with 16GB memory and a Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz with 12 cores. 
For the int8 simulations we use an RTX 2080 with 12GB memory.

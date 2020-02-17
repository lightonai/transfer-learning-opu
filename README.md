# TL_blogpost
Transfer Learning with the OPU - Blogpost code
## How to install

Clone the repository and then do `pip install <path_to_repo>`.

Should you wish to replicate the results with `TensorRT` in `int8` you need to download the appropriate version from the official 
website [here](https://developer.nvidia.com/tensorrt). We tested the code with `TensorRT 6.0.1.5` with `CUDA 10.1`.

`Pillow 6.1` is required to correctly run the code with TensorRT due to problems with the `.onnx` conversion. 
You can easily install it with `pip install Pillow==6.1`. 

Finally download the dataset from [here](https://www.kaggle.com/alessiocorrado99/animals10). You should put the dataset 
in the  same folder as the repo, but all scripts have an option to change the path with `-dataset_path` 

## Replicate the OPU/backprop results

Use the script `multiple_block.sh` in the `bash` folder. Open it in a text editor and then:
- set the OPU/backprop flags at the top to `true`, depending on which simulation you want to run
- Set the dtype to `float32`/`float16`. This affects only the OPU simulation.

- (optional) change the path to the script/dataset/save folder if you want to deviate from the 
defaults and then launch `./multiple_block.sh`. 

Finally execute the script with `./multiple_block.sh`. 
You might need to run `chmod +x multiple_block.sh` to make the script executable.
 
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

If you want to just extract the dataset features you can use the `tensorrt_extract_features.py` 
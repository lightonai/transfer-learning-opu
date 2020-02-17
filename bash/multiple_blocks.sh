#!/bin/bash

# Script for compression to iterate through multiple models.

# ============= Common parameters =============

#Set the respective flag to True to run the OPU/backprop simulation
OPU_simulation=true
backprop_simulation=true

generator_path=../scripts/generate_block_layer_pair.py
main_path=../scripts/OPU_training.py
main_bp_path=../scripts/backprop_training.py

save_path=../data/
txt_path="$save_path/dummy/"
dataset_path=../datasets/animals10/

batch_size=32
OPU="Saturn"

declare -a models=("densenet169")
model_options="full"

first_block=30
last_block=30       #The highest number of blocks is 36 from vgg19.

# ============= OPU parameters =============

n_components=2
dtype="float32"

encode_type='positive'

alpha_exp_min=6
alpha_exp_max=8
alpha_space=5

# ============= Backprop parameters =============

n_epochs=5
acc_toll=2

for model in "${models[@]}";
    do
    python3 $generator_path $model $first_block $last_block -save_path=$txt_path

    file=$txt_path"block_layer_pairs_$model.txt"

    while IFS="." read -r block layer; do
        case "$key" in '#'*) ;; *)

            if [[ "$OPU_simulation" == true ]]
            then
              python3 $main_path $model $dataset $OPU -model_options=$model_options -model_dtype=$dtype \
                      -block=$block -layer=$layer -n_components=$n_components -encode_type=$encode_type\
                      -alpha_exp_min=$alpha_exp_min -alpha_exp_max=$alpha_exp_max -alpha_space=$alpha_space \
                      -dataset_path=$dataset_path -save_path=$save_path
              echo -e '\n'
            fi

            if [[ "$backprop_simulation" == true ]]
            then
              python3 $main_bp_path $model $dataset Adam $OPU $n_epochs \
                      -block=$block -layer=$layer -dataset_path=$dataset_path -save_path=$save_path
              echo -e '\n'
            fi

        esac
    done < $file

    rm $file

    echo -e '\n'
done

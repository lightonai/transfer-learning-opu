import os
import pathlib

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tlopu.dataset import CatsAndDogs, CUB_200, Animals10, get_calibration_loader
from tlopu.model_utils import pick_model, get_model_size, save_model, cut_model
from tlopu.trt_utils import build_engine_onnx, calibrator, trt_conv_features

import pandas as pd
import numpy as np

from datetime import datetime

import argparse
from argparse import RawTextHelpFormatter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates and saves the convolutional features computed with TensorRT."
                    "Useful if the features have to be moved between multiple machines.",
        formatter_class=RawTextHelpFormatter)

    parser.add_argument("model_name",
                        help='Base model for TL. The script will load it from torchvision, save it as .onnx'
                             'in the specified path and load it in TensorRT', type=str)
    parser.add_argument("batch_size", help='Batch size for the training and testing phases. Defaults to 32.',
                        type=int, default=32)
    parser.add_argument("-dataset_name", help="Dataset name. defaults to 'animals10'", type=str, default="animals10")
    parser.add_argument("-cal_batch_size",
                        help='Only for int8. Batch size for the calibrator loader. '
                             'Can be different from the batch size for training and test. Defaults to 32.',
                        type=int, default=32)
    parser.add_argument("-n_cal_batches",
                        help='Only for int8. Number of batches for the calibrator loader. Defaults to 1.', type=int,
                        default=1)
    parser.add_argument("-num_workers", help="Number of workers. Defaults to 12", type=int, default=12)

    parser.add_argument('-model_options', help='Options for the removal of specific layers in the architecture.'
                                               'Defaults to full.',
                        choices=['full', 'noavgpool', 'norelu', 'norelu_maxpool'], type=str, default="full")

    parser.add_argument("-block", help='Index of the last block of the model. Defaults to 7.', type=int,
                        default=7)
    parser.add_argument("-layer", help='Index of the last layer of the last block of the model. Defaults to 4.',
                        type=int, default=4)

    parser.add_argument('-dtype_train', help="dtype for the training phase. Defaults to 'f32'.",
                        choices=['f32', 'f16', 'int8'], type=str, default="f32")
    parser.add_argument('-dtype_inf', help="dtype for the inference phase. Defaults to 'f32'.",
                        choices=['f32', 'f16', 'int8'], type=str, default="f32")

    parser.add_argument("-dataset_path",
                        help='Path to the dataset folder (excluded). Defaults to /data/home/luca/datasets/.', type=str,
                        default='/data/home/luca/datasets/')
    parser.add_argument("-onnx_path",
                        help='Path to the onnx folder where the model will be saved.'
                             'Defaults to /data/home/luca/quantization_trt/models/.',
                        type=str, default='/data/home/luca/quantization_trt/models/')

    parser.add_argument("-features_path",
                        help='Save path for the convolutional features. '
                             'Defaults to /data/home/luca/datasets_conv_features/int8_features.',
                        type=str, default='/data/home/luca/datasets_conv_features/int8_features')

    args = parser.parse_args()

    return args

def get_loaders(dataset_name, batch_size=32, num_workers=12, mean=None, std=None):
    transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if mean is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    data_transform = transforms.Compose(transform_list)

    if dataset_name == "cats_dogs":
        dataset_path = "/data/home/luca/datasets/cats-and-dogs-breeds-classification-oxford-dataset"

        train_dataset = CatsAndDogs(dataset_path, mode="trainval", transform=data_transform)
        test_dataset = CatsAndDogs(dataset_path, mode="test", transform=data_transform)

    elif dataset_name == "CUB_200":
        dataset_path = "/data/home/luca/datasets/CUB_200_2011/"

        train_dataset = CUB_200(dataset_path, mode="train", transform=data_transform)
        test_dataset = CUB_200(dataset_path, mode="test", transform=data_transform)


    elif dataset_name == "animals10":
        path = "/data/home/luca/datasets/animals10/raw-img/"
        train_dataset = Animals10(path, test_ratio=20, mode="train", transform=data_transform)
        test_dataset = Animals10(path, test_ratio=20, mode="test", transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader

def get_mean_std(train_loader):
    mean, std = torch.zeros(3), torch.zeros(3)

    for batch_id, (image, target) in enumerate(train_loader):
        mean += torch.mean(image, dim=(0, 2, 3))
        std += torch.std(image, dim=(0, 2, 3))

    mean = mean / len(train_loader)
    std = std / len(train_loader)

    return mean, std


def main(args):
    print('model = {}\tmodel options = {}\ndtype - train = {}\t dtype - inference = {}'
          .format(args.model_name, args.model_options, args.dtype_train, args.dtype_inf))

    train_loader, test_loader = get_loaders(args.dataset_name, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Computing dataset mean...")
    mean, std = get_mean_std(train_loader)
    train_loader, test_loader = get_loaders(args.dataset_name, batch_size=args.batch_size, num_workers=args.num_workers,
                                            mean=mean, std=std)
    print("Dataset = {}\ttrain images = {}\ttest images = {}\n"
          .format(args.dataset_name, len(train_loader.dataset), len(test_loader.dataset)))

    # Load the model.
    model, _ = pick_model(model_name=args.model_name, model_options=args.model_options, pretrained=True)

    model, output_size = cut_model(model, args.block, args.layer)
    print("Cut to block {} layer {}. Output size = {}".format(args.block, args.layer, output_size))

    model_size, total_weights, model_size_linear = get_model_size(model)
    print("model size = {0:3.2f} MB".format(model_size))

    onnx_path = os.path.join(args.onnx_path, args.model_name + ".onnx")
    save_model(model, args.batch_size, savepath=onnx_path)

    if args.dtype_train == 'int8' or args.dtype_inf == 'int8':
        print("Generating calibration loader...")
        calibrator_loader = get_calibration_loader(train_loader, cal_batch_size=args.cal_batch_size,
                                                   n_calibration_batches=args.n_cal_batches)

        cache_path = os.path.join(args.onnx_path, "cache")
        pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
        calib = calibrator(calibrator_loader,
                           os.path.join(cache_path, "animal10_{}_{}_{}.cache".format(args.model_name, args.model_options,
                                                                               args.n_cal_batches)))
    else:
        calib = None

    with build_engine_onnx(onnx_path, args.batch_size, max_workspace_size=4, dtype=args.dtype_train,
                           calibrator=calib) as engine:

        train_conv_features, train_labels, train_conv_time_full, train_conv_time = trt_conv_features(train_loader,
                                                                                                     engine,
                                                                                                     output_size)
        print("\nTraining - Total time in {0} : {1:3.3f} s (no data loading {2:3.3f} s)\n".format(args.dtype_train,
                                                                                                  train_conv_time_full,
                                                                                                  train_conv_time))

    with build_engine_onnx(onnx_path, args.batch_size, max_workspace_size=4, dtype=args.dtype_inf,
                           calibrator=calib) as engine:

        test_conv_features, test_labels, test_conv_time_full, test_conv_time = trt_conv_features(test_loader,
                                                                                                 engine,
                                                                                                 output_size)
        print("\nTest - Total time in {0} : {1:3.3f} s (no data loading {2:3.3f} s)\n".format(args.dtype_inf,
                                                                                              test_conv_time_full,
                                                                                              test_conv_time))
    current_date = str(datetime.now())

    path_to_features = os.path.join(args.features_path, "{}_{}.{}".format(args.model_name, args.block, args.layer),
                                    args.dataset_name)
    pathlib.Path(path_to_features).mkdir(parents=True, exist_ok=True)

    np.savez_compressed(os.path.join(path_to_features, "features.npz"),
                        train=train_conv_features, test=test_conv_features)
    np.savez_compressed(os.path.join(path_to_features, "labels.npz"), train=train_labels,
                        test=test_labels)

    data_columns = ["model", "model options", "dataset", 'batch size', 'batch size-cal', 'batch number-cal',
                    'dtype-train', 'dtype-test', 'conv f-train-full', 'conv f-train', 'conv f-test-full', 'conv f-test',
                    'date']

    data = [args.model_name, args.model_options, args.dataset_name, args.batch_size, args.cal_batch_size,
            args.n_cal_batches, args.dtype_train, args.dtype_inf, train_conv_time_full, train_conv_time,
            test_conv_time_full, test_conv_time, current_date]

    pd.DataFrame([data], columns=data_columns).to_csv(os.path.join(path_to_features, "data.csv"),
                                                      sep='\t', index=False)
    print("Features and csv data for {}-{} saved successfully in {}.\n".format(args.dataset_name, args.model_name,
                                                                             path_to_features))
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)

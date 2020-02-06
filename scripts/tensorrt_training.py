import os
import pathlib
import re
import argparse
from argparse import RawTextHelpFormatter
from time import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier

from tlopu.model_utils import pick_model, cut_model, get_model_size, save_model
from tlopu.features import encoding, decoding, get_random_features, dummy_predict_GPU
from tlopu.trt_utils import build_engine_onnx, calibrator, trt_conv_features
from tlopu.dataset import CatsAndDogs, CUB_200, Animals10, get_calibration_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Model compression on cut model - VGG, ResNet and DenseNet",
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument("model_name", help='Base model for TL.', type=str)
    parser.add_argument("OPU", help='OPU model. For file naming.', type=str, choices=['Zeus', 'Vulcain', "Saturn"])

    parser.add_argument("-device",
                        help="Device for the GPU computation, specified as 'cuda:x', where x is the GPU number."
                             "Choose 'cpu' to use the CPU for all computations. Defaults to 'cuda:0'", type=str,
                        default='cuda:0')
    parser.add_argument("-num_workers", help="Number of workers. Defaults to 12", type=int, default=12)
    parser.add_argument("-batch_size", help="Batch size. Defaults to 32", type=int, default=32)
    parser.add_argument("-cal_batch_size",
                        help='Only for int8. Batch size for the calibrator loader. '
                             'Can be different from the batch size for training and test. Defaults to 32.',
                        type=int, default=32)
    parser.add_argument("-n_cal_batches",
                        help='Only for int8. Number of batches for the calibrator loader. Defaults to 1.', type=int,
                        default=1)

    parser.add_argument('-model_options', help='Options for the removal of specific layers in the architecture.'
                                               'Defaults to full.',
                        choices=['full', 'noavgpool', 'norelu', 'norelu_maxpool'], type=str, default="full")
    parser.add_argument('-dtype_train', help="dtype for the training phase. Defaults to 'f32'.",
                        choices=['f32', 'f16', 'int8'], type=str, default="f32")
    parser.add_argument('-dtype_inf', help="dtype for the inference phase. Defaults to 'f32'.",
                        choices=['f32', 'f16', 'int8'], type=str, default="f32")

    parser.add_argument("-encode_type",
                        help="Type of encoding. 'float32'=standard float32 | 'abs_th'= 1bit float32-like | "
                             "'plain_th'=Threshold encoding",
                        type=str, choices=['float32', 'abs_th', 'plain_th'], default='float32')
    parser.add_argument("-encode_thr",
                        help="Threshold for the 'abs_th' and 'plain_th' encoding. Defaults to 2.", type=int, default=2)

    parser.add_argument("-decode_type", help='Type of decoding. Defaults to mixing', type=str,
                        choices=['none', 'mixing'],
                        default='mixing')
    parser.add_argument("-exp_bits", help='Number of bits for encoding and decoding. Defaults to 1', type=int,
                        default=1)

    parser.add_argument("-block", help='Index of the last block of the model. Defaults to 7.', type=int,
                        default=7)
    parser.add_argument("-layer", help='Index of the last layer of the last block of the model. Defaults to 4.',
                        type=int,
                        default=4)

    parser.add_argument("-n_components", help='Number of random features. Defaults to 200000.', type=int,
                        default=200000)

    parser.add_argument("-alpha_exp_min",
                        help='Minimum order of magnitude for the regularization coefficient. Defaults to 3.', type=int,
                        default=3)
    parser.add_argument("-alpha_exp_max",
                        help='Maximum order of magnitude for the regularization coefficient.Defaults to 5.', type=int,
                        default=5)
    parser.add_argument("-alpha_space",
                        help='Spacing between the mantissa of the regularization coefficients. Defaults to 5.',
                        type=int,
                        default=5)

    parser.add_argument("-dataset_path", help='Path to the dataset folder (excluded).', type=str,
                        default='/data/home/luca/datasets/cats-and-dogs-breeds-classification-oxford-dataset')
    parser.add_argument("-onnx_path",
                        help='Path to the onnx folder where the model will be saved. '
                             'Defaults to /data/home/luca/quantization_trt/models/.',
                        type=str, default='/data/home/luca/quantization_trt/models/')

    parser.add_argument("-save_path",
                        help='Path to the save folder. If None, results will not be saved. Defaults to None.',
                        type=str, default=None)

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
    print('model = {}\tmodel options = {}'.format(args.model_name, args.model_options))

    train_loader, test_loader = get_loaders("animals10", batch_size=args.batch_size, num_workers=args.num_workers)
    print("Computing dataset mean...")
    mean, std = get_mean_std(train_loader)
    train_loader, test_loader = get_loaders("animals10", batch_size=args.batch_size, num_workers=args.num_workers,
                                            mean=mean, std=std)

    print("train images = {}\ttest images = {}".format(len(train_loader.dataset), len(test_loader.dataset)))

    alpha_mant = np.linspace(1, 9, args.alpha_space)
    alphas = np.concatenate([alpha_mant * 10 ** i for i in range(args.alpha_exp_min, args.alpha_exp_max + 1)])

    if args.save_path is not None:
        base_path = os.path.join(args.save_path, '{}_{}_brutal'.format(args.model_name, args.OPU),
                                 'OPU_{}_{}_{}'.format(args.n_components, args.model_options, args.dtype_train))

        pathlib.Path(os.path.join(base_path, 'train')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(base_path, 'inference')).mkdir(parents=True, exist_ok=True)

        train_data_columns = ['RP', 'block', 'layer', 'model dtype', "conv full", 'conv f', 'conv features shape',
                              args.encode_type, 'rand f', 'to float32', 'alpha', 'fit time', 'total time',
                              'acc_test', 'model size [MB]', 'total weights', 'ridge size [MB]', 'date']

        inference_data_columns = ['RP', 'block', 'layer', 'model dtype', "conv full", 'conv f', args.encode_type,
                                  'rand f', 'to float32', 'predict time', 'inference time', 'model size [MB]',
                                  'ridge size [MB]']

    # Load the model.
    model, _ = pick_model(model_name=args.model_name, model_options=args.model_options, pretrained=True)

    model, output_size = cut_model(model, args.block, args.layer)
    print("Cut to block {} layer {}. Output size = {}".format(args.block, args.layer, output_size))

    model_size, total_weights, model_size_linear = get_model_size(model)
    print("model size = {0:3.2f} MB".format(model_size))

    if args.n_components != 0:
        args.n_components = output_size // args.n_components
        print("Random projection from {} to {}".format(output_size, args.n_components))

    onnx_path = os.path.join(args.onnx_path, args.model_name + ".onnx")
    save_model(model, args.batch_size, savepath=onnx_path)

    if args.dtype_train == 'int8' or args.dtype_inf == 'int8':
        print("Generating calibration loader...")
        calibrator_loader = get_calibration_loader(train_loader, cal_batch_size=args.cal_batch_size,
                                                   n_calibration_batches=args.n_cal_batches)

        cache_path = os.path.join(args.onnx_path, "cache")
        pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
        calib = calibrator(calibrator_loader,
                           os.path.join(cache_path, "{}_{}_{}.cache".format(args.model_name, args.model_options,
                                                                            args.n_cal_batches)))
    else:
        calib = None

    with build_engine_onnx(onnx_path, args.batch_size, max_workspace_size=4, dtype=args.dtype_train,
                           calibrator=calib) as engine:

        train_conv_features, train_labels, train_conv_time_full, train_conv_time = trt_conv_features(train_loader,
                                                                                                     engine,
                                                                                                     output_size)
        print("\nTraining - Total time in {0} : {1:2.3f} s (no data loading {2:2.3f} s)\n".format(args.dtype_train,
                                                                                                  train_conv_time_full,
                                                                                                  train_conv_time))

        test_conv_features, test_labels, test_conv_time_full, test_conv_time = trt_conv_features(test_loader,
                                                                                                 engine,
                                                                                                 output_size)
        print("\nTest - Total time in {0} : {1:2.3f} s (no data loading {2:2.3f} s)\n".format(args.dtype_inf,
                                                                                              test_conv_time_full,
                                                                                              test_conv_time))

    # Encode, get the random features and decode
    train_encode_time, test_encode_time, enc_train_features, enc_test_features = encoding(train_conv_features,
                                                                                          test_conv_features,
                                                                                          encode_type=args.encode_type,
                                                                                          threshold=args.encode_thr)

    train_proj_time, train_random_features = get_random_features(enc_train_features, args.n_components)
    test_proj_time, test_random_features = get_random_features(enc_test_features, args.n_components)

    del model, enc_train_features, enc_test_features

    train_decode_time, dec_train_random_features = decoding(train_random_features, decode_type=None)
    test_decode_time, dec_test_random_features = decoding(test_random_features, decode_type=None)

    print(
        "Train projection time = {0:3.2f} s\tTrain decode time = {1:3.2f} s".format(train_proj_time, train_decode_time))
    print("Test projection time = {0:3.2f} s\tTest decode time = {1:3.2f} s".format(test_proj_time, test_decode_time))
    torch.cuda.empty_cache()

    current_date = str(datetime.now())
    final_train_data = []
    final_inference_data = []

    # Run the ridge classifier

    for alpha in alphas:
        clf = RidgeClassifier(alpha=alpha)
        since = time()
        clf.fit(dec_train_random_features, train_labels)
        fit_time = time() - since

        train_accuracy = clf.score(dec_train_random_features, train_labels) * 100
        test_accuracy = clf.score(dec_test_random_features, test_labels) * 100

        test_decode_time, predict_time = dummy_predict_GPU(clf, dec_test_random_features, device=args.device)

        total_train_time = train_conv_time + train_encode_time + train_proj_time + train_decode_time + fit_time

        total_inference_time = test_conv_time + test_encode_time + test_proj_time + test_decode_time + predict_time

        n_bits = int(re.findall("\d+", args.dtype_train)[0])
        ridge_size = np.prod(clf.coef_.shape) * n_bits / (8 * 2 ** 10 * 2 ** 10)

        train_data = [args.n_components, args.block, args.layer, args.dtype_train, train_conv_time_full,
                      train_conv_time, output_size, train_encode_time, train_proj_time, train_decode_time, alpha,
                      fit_time, total_train_time, test_accuracy, model_size, total_weights, ridge_size, current_date]

        inference_data = [args.n_components, args.block, args.layer, args.dtype_inf, test_conv_time_full,
                          test_conv_time, test_encode_time, test_proj_time, test_decode_time, predict_time,
                          total_inference_time, model_size, ridge_size]

        final_train_data.append(train_data)
        final_inference_data.append(inference_data)

        print('alpha = {0:.2e}\ttrain acc = {1:3.2f}\ttest acc = {2:2.2f}\tInference time = {3:3.2f} s'
              .format(alpha, train_accuracy, test_accuracy, total_inference_time))

    if args.save_path is not None:
        csv_name_train = os.path.join("train", "{}_{}_{}.{}_train.csv".format(args.model_name, args.model_options,
                                                                              str(args.block).zfill(2),
                                                                              str(args.layer).zfill(2)))
        csv_name_inference = os.path.join("inference", "{}_{}_{}.{}_inference.csv".format(args.model_name,
                                                                                          args.model_options,
                                                                                          str(args.block).zfill(2),
                                                                                          str(args.layer).zfill(2)))

        pd.DataFrame(final_train_data, columns=train_data_columns).to_csv(
            os.path.join(base_path, csv_name_train), sep='\t', index=False)

        pd.DataFrame(final_inference_data, columns=inference_data_columns).to_csv(
            os.path.join(base_path, csv_name_inference), sep='\t', index=False)
        print("Results saved in ", base_path)
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)

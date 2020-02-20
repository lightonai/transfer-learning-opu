import os
import pathlib
import argparse
from argparse import RawTextHelpFormatter
from time import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pandas as pd

from tqdm import tqdm

from tlopu.model_utils import pick_model, Reshape, cut_model, get_model_size
from tlopu.backprop import train_model, evaluate_model
from tlopu.dataset import Animals10

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer learning - standard", formatter_class=RawTextHelpFormatter)

    parser.add_argument("model_name", help='Base model for TL.', type=str)
    parser.add_argument("optimizer_name", help='Choice of the optimizer.', type=str, choices=['Adam', 'SGD'])
    parser.add_argument("OPU", help='OPU model. For file naming.', type=str, choices=['Zeus', 'Vulcain', "Saturn"])

    parser.add_argument("epochs", help='Number of training epochs.', type=int)
    parser.add_argument("-num_workers", help="Number of workers. Defaults to 12", type=int, default=12)
    parser.add_argument("-batch_size", help="Batch size in the train and inference phase. Defaults to 32.",
                        type=int, default=32)

    parser.add_argument("-device",
                        help="Device for the GPU computation, specified as 'cuda:x', where x is the GPU number."
                             "Choose 'cpu' to use the CPU for all computations. Defaults to 'cuda:0'", type=str,
                        default='cuda:0')
    parser.add_argument('-model_options', help='Options for the removal of layers in the architecture',
                        choices=['full', 'noavgpool', 'norelu', 'norelu_maxpool'], default='full')

    parser.add_argument("-lr", help='Learning rate. Defaults to 0.000001', type=float, default=0.000001)

    parser.add_argument("-block", help='Index of the last block of the model. Defaults to 7.', type=int,
                        default=7)
    parser.add_argument("-layer", help='Index of the last layer of the last block of the model. Defaults to 4.',
                        type=int, default=4)

    parser.add_argument("-dataset_path", help='Path to the dataset folder (excluded).', type=str,
                        default='../datasets/')
    parser.add_argument("-save_path",
                        help='Path to the save folder. If None, results will not be saved. Defaults to None.',
                        type=str, default=None)

    args = parser.parse_args()

    return args


def get_loaders(dataset_path, batch_size=32, num_workers=12, mean=None, std=None):
    """
    Function to load the train/test loaders.

    Parameters
    ----------
    dataset_path: str, dataset path.

    batch_size: int, batch size.
    num_workers: int, number of workers.
    mean:None or torch.Tensor, mean per channel
    std:None or torch.Tensor, std per channel

    Returns
    -------
    train_loader: Pytorch dataloader, dataloader for the train set.
    test_loader: Pytorch dataloader, dataloader for the test set.
    """


    transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if mean is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    data_transform = transforms.Compose(transform_list)

    dataset_path = os.path.join(dataset_path, "animals10/raw-img/")

    train_dataset = Animals10(dataset_path, test_ratio=20, mode="train", transform=data_transform)
    test_dataset = Animals10(dataset_path, test_ratio=20, mode="test", transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


def get_mean_std(train_loader):
    """
    Computes the mean and std per channel on the train dataset.

    Parameters
    ----------
    train_loader: Pytorch dataloader, dataloader for the train set

    Returns
    -------
    mean: torch.Tensor, mean per channel
    std: torch.Tensor, std per channel
    """

    mean, std = torch.zeros(3), torch.zeros(3)

    for batch_id, (image, target) in enumerate(train_loader):
        mean += torch.mean(image, dim=(0, 2, 3))
        std += torch.std(image, dim=(0, 2, 3))

    mean = mean / len(train_loader)
    std = std / len(train_loader)

    return mean, std

def save_data(args, final_data):
    """
    Helper function to save the relevant simulation data in the correct folder.

    Parameters
    ----------
    args: argparse arguments. Used to ensure that the folder structure is unique for each simulation.
    final_data: list, contains the relevant simulation data.

    Returns
    -------

    """

    base_path = os.path.join(args.save_path, '{}_{}'.format(args.model_name, args.OPU), "backprop")

    pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)

    final_data_columns = ['block', 'layer', 'conv features shape', 'optimizer', 'epoch', 'train loss',
                          'test loss', 'test accuracy', 'training time', "test time", "model size-linear",
                          "model size-tot", 'date']
    csv_name = "{}_{}_backprop_{}.{}.csv".format(args.model_name, args.model_options,
                                                 str(args.block).zfill(2), str(args.layer).zfill(2))

    pd.DataFrame(final_data, columns=final_data_columns).to_csv(os.path.join(base_path, csv_name),
                                                                sep='\t', index=False)

    return

def main(args):
    criterion_backprop = nn.CrossEntropyLoss(reduction='sum')

    print('model = {}\tmodel options = {}\n'.format(args.model_name, args.model_options))

    train_loader, test_loader = get_loaders(args.dataset_path, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Computing dataset mean...")
    mean, std = get_mean_std(train_loader)
    train_loader, test_loader = get_loaders(args.dataset_path, batch_size=args.batch_size,
                                            num_workers=args.num_workers, mean=mean, std=std)

    print("train images = {}\ttest images = {}\n".format(len(train_loader.dataset), len(test_loader.dataset)))

    model, output_size = pick_model(model_name=args.model_name, model_options=args.model_options, pretrained=True)

    model, new_output_size = cut_model(model, args.block, args.layer)
    print("Cut to block {} layer {}. Output size = {}".format(args.block, args.layer, new_output_size))

    model.reshape = Reshape()
    model.classifier = nn.Linear(new_output_size, len(train_loader.dataset.classes))
    model_size_tot, total_weights, model_size_linear = get_model_size(model)
    print("model size = {0:3.2f} MB".format(model_size_tot))

    if args.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        print('Optimizer not recognized.')

    model.to(torch.device(args.device))
    final_data = []

    for epoch in tqdm(range(args.epochs)):
        # Training phase
        torch.cuda.synchronize()
        t0 = time()

        model, epoch_loss, epoch_acc = train_model(model, train_loader, criterion_backprop, optimizer, device=args.device)

        torch.cuda.synchronize()
        training_time = time() - t0

        # Test phase
        torch.cuda.synchronize()
        t0 = time()

        test_loss, test_acc, inference_conv_time, inference_linear_time = evaluate_model(model, test_loader,
                                                                                         criterion_backprop,
                                                                                         optimizer, device=args.device)

        torch.cuda.synchronize()
        test_time = time() - t0

        print('Train Loss {:.4f}\tTrain acc: {:3.2f}\tTrain RT = {:3.2f}\tTest Loss {:.4f}\tTest acc: {:3.2f}\tTest RT = {:3.2f}'
              .format(epoch_loss, epoch_acc, training_time, test_loss, test_acc, test_time))

        # Save data
        epoch_data = [args.block, args.layer, new_output_size, args.optimizer_name, epoch, epoch_loss, test_loss,
                      test_acc, training_time, test_time, model_size_linear, model_size_tot, datetime.now()]

        final_data.append(epoch_data)

        if args.save_path is not None:
            save_data(args, final_data)

    del model
    torch.cuda.empty_cache()
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)

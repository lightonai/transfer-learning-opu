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

from tlopu.model_utils import pick_model, Reshape, cut_model, get_model_size
from tlopu.backprop import train_model, evaluate_model
from tlopu.dataset import CatsAndDogs

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

    parser.add_argument("-lr", help='Learning rate. Defaults to 0.001', type=float, default=0.001)
    parser.add_argument("-acc_toll", help='Tollerance threshold on the train accuracy in percentage. Defaults to 2',
                        type=int, default=2)

    parser.add_argument("-block", help='Index of the last block of the model. Defaults to 7.', type=int,
                        default=7)
    parser.add_argument("-layer", help='Index of the last layer of the last block of the model. Defaults to 4.',
                        type=int,
                        default=4)

    parser.add_argument("-dataset_path", help='Path to the dataset folder (excluded).', type=str,
                        default='/data/home/luca/datasets/cats-and-dogs-breeds-classification-oxford-dataset')
    parser.add_argument("-save_path",
                        help='Path to the save folder. If None, results will not be saved. Defaults to None.',
                        type=str, default=None)

    args = parser.parse_args()

    return args


def main(args):
    acc_toll = args.acc_toll / 100
    criterion_backprop = nn.CrossEntropyLoss(reduction='sum')

    print('model = {}\tmodel options = {}\n'.format(args.model_name, args.model_options))

    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_dataset = CatsAndDogs(args.dataset_path, mode="trainval", transform=train_transform)
    test_dataset = CatsAndDogs(args.dataset_path, mode="test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print("train images = {}\ttest images = {}\n".format(len(train_loader.dataset), len(test_loader.dataset)))

    if args.save_path is not None:
        base_path = os.path.join(args.save_path, '{}_{}_brutal'.format(args.model_name, args.OPU),"backprop")

        pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)

        final_data_columns = ['block', 'layer', 'conv features shape', 'optimizer', 'train epochs', 'train loss',
                              'test loss',
                              'test accuracy', 'training time', "test time", "model size-linear", "model size-tot",
                              'date']

    model, output_size = pick_model(model_name=args.model_name, model_options=args.model_options, pretrained=True)

    model, new_output_size = cut_model(model, args.block, args.layer)
    print("Cut to block {} layer {}. Output size = {}".format(args.block, args.layer, new_output_size))

    model.reshape = Reshape()
    model.classifier = nn.Linear(new_output_size, len(train_loader.dataset.classes))
    model_size_tot, total_weights, model_size_linear = get_model_size(model)
    print("model size = {0:3.2f} MB".format(model_size_tot))

    if args.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    elif args.optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.lr, momentum=0.9)
    else:
        print('Optimizer not recognized.')

    model.to(torch.device(args.device))

    # Training phase
    torch.cuda.synchronize()
    t0 = time()

    model, loss_list, train_acc_list, stop_epoch = train_model(model, args.epochs, train_loader, criterion_backprop,
                                                               optimizer, acc_toll=acc_toll, device=args.device)

    torch.cuda.synchronize()
    training_time = time() - t0
    print('Training time = ', training_time)

    # Test phase
    torch.cuda.synchronize()
    t0 = time()

    test_loss, test_acc, inference_conv_time, inference_linear_time = evaluate_model(model, test_loader,
                                                                                     criterion_backprop,
                                                                                     optimizer, device=args.device)

    torch.cuda.synchronize()
    test_prop = time() - t0
    print('Loss {:.4f} \t Test acc: {:.4f} \t Test RT = {:.4f}'.format(test_loss, test_acc, test_prop))

    # Save data
    final_data = [args.block, args.layer, new_output_size, args.optimizer_name, stop_epoch, loss_list[-1], test_loss,
                  test_acc, training_time, test_prop, model_size_linear, model_size_tot, datetime.now()]

    if args.save_path is not None:
        csv_name = "{}_{}_backprop_{}.{}.csv".format(args.model_name, args.model_options,
                                                     str(args.block).zfill(2), str(args.layer).zfill(2))

        pd.DataFrame([final_data], columns=final_data_columns).to_csv(os.path.join(base_path, csv_name),
                                                                      sep='\t', index=False)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    main(args)

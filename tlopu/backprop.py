from tqdm import tqdm
from time import time
import numpy as np
import torch


def train_model(model, epochs, train_loader, criterion, optimizer, acc_toll, device='cpu'):
    """
    model: Pytorch model,
        neural net model.
    epochs: int,
        number of training epochs.
    train_loader: torch Dataloader,
        contains the training images.
    criterion: torch.nn.modules.loss,
        criterion for the determination of the loss.
    acc_toll: float,
        tollerance on the train accuracy. If the difference between two consecutive epochs goes below this,
        stop the training.
    device: string,
        device to use for the computation. Choose between 'cpu' and 'gpu:x', where
        x is the GPU number. Defaults to 'cpu'.

    Returns
    -------
    train_loss: float,
        loss on the test set at each training epoch.
    train_acc: float,
        accuracy on the test set [%] at each training epoch.
    """

    tot_train_images = len(train_loader.dataset)

    loss_list, train_acc_list = [], []
    model.train()
    for epoch in tqdm(range(epochs)):

        running_loss = 0.0
        running_corrects = 0

        for batch_id, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            images = images.to(torch.device(device))
            labels = labels.to(torch.device(device))

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.data
            running_corrects += torch.sum(preds == labels.data)
            torch.cuda.synchronize()

            del images, outputs

        epoch_loss = running_loss.item() / tot_train_images
        epoch_acc = running_corrects.item() / tot_train_images

        loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)

        print('Epoch = {} \t Loss = {:.4f} \t Train acc = {:.4f}'.format(epoch, epoch_loss, epoch_acc))

        stop_epoch = epoch + 1
        if epoch != 0 and np.abs(train_acc_list[-1] - train_acc_list[-2]) < acc_toll:
            break

    return model, loss_list, train_acc_list, stop_epoch


def evaluate_model(model, test_loader, criterion, dtype='f32', device='cpu'):
    """

    model: Pytorch model,
        neural net model.
    test_loader: torch Dataloader,
        contains the test images.
    criterion: torch.nn.modules.loss,
        criterion for the determination of the loss. Defaults to CrossEntropyLoss.
    device: string,
        device to use for the computation. Choose between 'cpu' and 'gpu:x', where
        x is the GPU number. Defaults to 'cpu'.

    Returns
    -------
    test_loss: float,
        loss on the test set.
    test_acc: float,
        accuracy on the test set [%].
    inference_full: float,
        inference time, including the data loading.
    inference_conv_time: float,
        inference time for the convolutional part only.
    inference_linear_time: float,
        inference time for the linear part only.
    """
    tot_test_images = len(test_loader.dataset)

    running_loss = 0.0
    running_corrects = 0
    inference_conv_time = 0

    if dtype == 'f16':
        model.half()

    model.to(torch.device(device)).eval()

    torch.cuda.synchronize()
    full_start = time()

    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(test_loader):

            images = images.to(torch.device(device))
            labels = labels.to(torch.device(device))

            if dtype == 'f16':
                images = images.half()

            torch.cuda.synchronize()
            t0 = time()

            outputs = model(images)

            torch.cuda.synchronize()
            inference_conv_time += time() - t0

            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.data
            running_corrects += torch.sum(preds == labels.data)

            del images, outputs

        test_loss = running_loss.item() / tot_test_images
        test_acc = running_corrects.item() / tot_test_images * 100

    torch.cuda.synchronize()
    inference_full = time() - full_start

    return test_loss, test_acc, inference_full, inference_conv_time

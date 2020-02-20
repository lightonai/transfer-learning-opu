from time import time
import torch


def train_model(model, train_loader, criterion, optimizer, device='cpu'):
    """
    Trains the given model for one epoch on the given dataset.

    Parameters
    ----------

    model: Pytorch model,
        neural net model.
    train_loader: torch Dataloader,
        contains the training images.
    criterion: torch.nn.modules.loss,
        criterion for the determination of the loss.
    optimizer: torch.optim,
        optimizer for the training.
    device: string,
        device to use for the computation. Choose between 'cpu' and 'gpu:x', where
        x is the GPU number. Defaults to 'cpu'.

    Returns
    -------
    model: torch model,
        model trained on the given dataset.
    epoch_loss: float,
        loss on the train set.
    epoch_acc: float,
        accuracy on the train set [%].
    """

    tot_train_images = len(train_loader.dataset)

    model.train()

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
    epoch_acc = running_corrects.item() / tot_train_images * 100

    return model, epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, dtype='float32', device='cpu'):
    """

    model: Pytorch model,
        neural net model.
    test_loader: torch Dataloader,
        contains the test images.
    criterion: torch.nn.modules.loss,
        criterion for the determination of the loss. Defaults to CrossEntropyLoss.
    dtype: str,
        dtype for the inference. Choose between float32 and float16.
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
    """
    tot_test_images = len(test_loader.dataset)

    running_loss = 0.0
    running_corrects = 0
    inference_conv_time = 0

    if dtype == 'float16':
        model.half()

    model.to(torch.device(device)).eval()

    torch.cuda.synchronize()
    full_start = time()

    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(test_loader):

            images = images.to(torch.device(device))
            labels = labels.to(torch.device(device))

            if dtype == 'float16':
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

import os
import pathlib
from warnings import warn

import torch
import itertools

import torchvision.models as models
from torch.nn import Linear, Sequential

class Reshape(torch.nn.Module):
    """
    Layer used to reshape the output of a convolutional from a 4D tensor (batch_size, C, H, W)
    to a 2D one (batch_size, -1). Needed when using 2 or more GPUs, as DataParallel objects are not subscriptable.
    """

    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def pick_model(model_name, model_options='full', pretrained=True, device='cpu', input_shape=(1, 3, 224, 224),
               linear_out=0, dtype='float32'):
    """
    Loads the desired model from the torchvision.models module.

    Parameters
    ----------
    model_name: string,
        name of the model. Needs to correspond to one of the attributes of the torchvsion package.
        More info here: https://pytorch.org/docs/stable/torchvision/models.html

    model_options: string,
        variations on the original model. Options are:
        - "norelu": removes the last ReLU in the vgg architectures;
        - "norelu_maxpool": removes the ReLU and maxpool layers at the end of the VGG architectures;
        - "noavgpool": removes the avgpool at the end of the ResNet architectures

    pretrained: boolean,
        if True, loads the pretrained weights of the model. Defaults to True.

    input_shape: tuple of int,
        shape of the input samples fed to the network in the form (batch_size, channels, heightm width).
        Defaults to (1,3,224,224).

    device: string,
        device for the gradient computation. Choose between "cpu" and "gpu:x", where x is the number of the GPU device.

    linear_out: int,
        number of classes of the dataset, needed in order to define the linear layer output.
        If 0, no linear layer is added. Defaults to 0.

    dtype: string,
        datatype for the model. Choose between 'float32' and 'float16'. Defaults to 'float32'.

    Returns
    -------
    model pytorch model,
        neural network model.
    output_size: int,
        size of the convolutional features in output to the model.

    NOTE: for ease of implementation in the code, the models are turned into a Sequential object,
        which erases the name of the network. An attribute 'model.name' has been added to fix this.
    """
    all_models = ["vgg16", "vgg19",
                  "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                  "densenet121", "densenet169", "densenet201"]


    assert model_name in all_models, "Model name should be one of {}".format(all_models)

    if model_name[:3] == 'vgg':
        model = getattr(models, model_name)(pretrained=pretrained).features

        if model_options == 'norelu':
            del model[-2]
        elif model_options == 'norelu_maxpool':
            del model[-2], model[-1]

    elif model_name[:6] == 'resnet':

        if model_options == 'noavgpool':
            layers = [i for i in range(8)]

        else:
            layers = [i for i in range(9)]

        model = getattr(models, model_name)(pretrained=pretrained)
        final_layers = [list(model.children())[i] for i in layers]
        model = Sequential(*final_layers)

    elif model_name[:8] == "densenet":
        model = getattr(models, model_name)(pretrained=pretrained).features

    else:
        model = EfficientNet.from_pretrained(model_name)

    model.name = model_name
    output_size = get_output_size(model, input_shape, device="cpu")

    if linear_out != 0:
        model.reshape = Reshape()
        model.classifier = Linear(output_size, linear_out, bias=True)

    model.to(torch.device(device))

    if dtype == 'float16':
        model.half()

    print('{} model loaded successfully.'.format(model_name))

    return model, output_size


def get_output_size(model, input_shape=(1, 3, 224, 224), device="cpu", dtype='float32'):
    """
    Returns the shape of the convolutional features in output to the model.

    Parameters
    ----------
    model pytorch model,
        neural network model.
    input_shape: tuple of int,
        shape of the images in input to the model in the form (batch_size, channels, height, width).
        Defaults to (1, 3, 224, 224).

    Return
    ------
    output_size : int,
        shape of the flattened convolutional features in output to the model.

    Note: It si not possible to do model(x) on CPU with f16. To avoid problems, the model is cast to float32 for this
    computation and then it is converted back to float16.
    """

    if dtype == "float16":
        model.float()

    dummy_input = torch.ones(input_shape).to(device)

    if model.name[0:12] == "efficientnet":
        output_size = model.extract_features(dummy_input).shape[1:].numel()
    else:
        output_size = model(dummy_input).shape[1:].numel()

    if dtype == "float16":
        model.half()

    return output_size


def generate_iterator(model, first_block, last_block):
    """
    Generates an iterator to loop through all the possible outputs of a DenseNet/ResNet.

    Parameters
    ----------
    model: pytorch model,
        pytorch model of a densenet/resnet.
    first_block: int,
        first block of the architecture to keep.
    last_block: int,
        last block of the architecture to keep. If the block is above the total number of layers, it is ignored.

    Returns
    -------
    full_iterator: iterator object,
        iterator object containing the (block, layer) indeces of the cut models.

    NOTE: the loop will start from the last blocks in the architecture and work its way up.
        This was done because the outputs of the first blocks are really big, and so might
        not fit in the GPU memory.
    """

    blocks = reversed([i for i in range(first_block, last_block + 1)])
    full_iterator = []

    single_blocks = (models.densenet._Transition, torch.nn.Conv2d, torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d,
                     torch.nn.MaxPool2d, torch.nn.ReLU, torch.nn.BatchNorm2d)

    for block in blocks:

        if block >= len(model):
            continue

        final_block = model[block]

        if isinstance(final_block, single_blocks):
            full_iterator.append(itertools.product([block], [0]))
        else:
            block_depth = len(final_block)
            full_iterator.append(itertools.product([block], reversed(range(block_depth))))

    full_iterator = itertools.chain.from_iterable(full_iterator)
    return full_iterator


def save_model(model, batch_size, savepath, channels=3):
    """
    Saves a model in the .onnx format.

    Parameters
    ----------
    model: Pytorch model,
        neural network.
    model_name:
    name of the model. For file naming

    Returns
    -------
    Nothing, but saves the model in the .onnx format in the specified folder.

    Note: Due to lack of implementation of some layers in float16, the model is converted in float32 beforehand.

    """

    print("Saving model to: ", savepath)
    if model[0].weight.data.dtype != torch.float32:
        model.type(torch.float32)

    input_size = (batch_size, channels, 224, 224)
    dummy_input = torch.ones(input_size)

    directory, file = os.path.split(savepath)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    torch.onnx.export(model, dummy_input, savepath, verbose=False)
    print("Model converted to .onnx successfully")

    return

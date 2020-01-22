from time import time

import torch
import numpy as np

from lightonml.encoding.base import Float32Encoder, MixingBitPlanDecoder
from lightonopu.opu import OPU


def fast_conv_features(loader, model, out_shape, encode_type='positive', dtype='float32', device='cpu'):
    """
    Computes the convolutional features of the images in the loader, and optionally encodes them on GPU
    based on their sign.

    Parameters
    ----------

    loader: torch Dataloader,
        contains the images to extract the training features from.
    model: torchvision.models,
        architecture to use to get the convolutional features.
    out_shape: int,
        output size of the last layer.
    n_images: int,
        number of images.
    encode: str,
        encodes the convolutional features on GPU. Choices: positive -> sign encoding / float32 -> 1 bit float32.
        Eventual other options result in no encoding.
    dtype: string,
        datatype for the computation of the convolutional features. Choose between 'float32' (default)
        and 'float16'. Make sure that the model in input matches the dtype.
    device: string,
        device to use for the computation. Choose between 'cpu' and 'gpu:x', where
        x is the GPU number. Defaults to 'cpu'.

    Returns
    -------

    conv_features: numpy array,
        array containing the convolutional features. format is (# of samples * # of features).
        They are already moved to CPU.
    labels: list of int,
        labels associated to each image.
    conv_time: float,
        time required to compute the convolutional features. It includes the data loading.
    encode_time: float,
        encoding time, done on the GPU.
    """

    n_images = len(loader.dataset)

    model.eval()
    batch_size = loader.batch_size
    torch.cuda.synchronize()
    t0 = time()

    conv_features = torch.cuda.FloatTensor(n_images, out_shape)
    labels = np.empty(n_images, dtype='uint8')

    if dtype == 'float16':
        conv_features = conv_features.half()

    for i, (images, targets) in enumerate(loader):

        images = images.to(torch.device(device))
        if dtype == 'float16':
            images = images.half()

        if model.name[0:12] == "efficientnet":
            outputs = model.extract_features(images)
        else:
            outputs = model(images)

        stop_idx = (i + 1) * batch_size
        if stop_idx > conv_features.shape[0]:
            stop_idx = conv_features.shape[0]

        conv_features[i * batch_size: (i + 1) * batch_size, :] = outputs.data.view(images.size(0), -1)
        labels[i * batch_size: (i + 1) * batch_size] = targets.numpy()

        del images, outputs

    torch.cuda.synchronize()
    conv_time = time() - t0

    if encode_type == 'positive':
        torch.cuda.synchronize()
        start = time()
        conv_features = (conv_features > 0)
        torch.cuda.synchronize()
        encode_time = time() - start

    elif encode_type == 'float32':
        torch.cuda.synchronize()
        start = time()
        conv_features = (torch.abs(conv_features) > 2)
        torch.cuda.synchronize()
        encode_time = time() - start

    else:
        encode_time = 0

    conv_features = conv_features.cpu().numpy()

    return conv_features, labels, conv_time, encode_time


def encoding(train_conv_features, test_conv_features, exp_bits=1, sign_bit=False, mantissa_bits=0,
             encode_type='float32', threshold=0):
    """
    Encodes the convolutional features. Three schemes are available:
    - float32 = Standard float32 encoding with a given number of bits
    - abs_th = 1 bit float32-like encoding: if abs(x)>th -> 1; else 0.
        abs_th with threshold 1 is 1bit float32, but much faster.
    - plain_th = Threshold encoding: if x>th -> 1; else 0
        plain_th with threshold 0 is a sign encoder.

    Parameters
    ----------

    train_conv_features: numpy 2d array,
        convolutional features of the training set.
    test_conv_features: numpy 2d array,
        convolutional features of the test set.
    exp_bits: int,
        number of bits for the exponent in the float32 encoding. Defaults to 1.
    sign_bit: boolean,
        if True, a bit will be used for the sign of the encoded numbers in the float32 encoding. Defaults to False.
    mantissa_bits: int,
        number of bits for the mantissa in the float32 encoding. Defaults to 0.
    encode type: string,
        type of encoding to perform. Choices = 'float32', 'abs_th' ,'plain_th'. Defaults to 'float32'.
    threshold: float,
        threshold for the abs_th/plain_th encodings. Defaults to 0.

    Returns
    -------
    train_encode_time: float,
        train encoding time.
    test_encode_time: float,
        test encoding time.
    encoded_train_conv_features: numpy 2d array,
        encoded convolutional training features.
    encoded_test_conv_features: numpy 2d array,
        encoded convolutional test features.

    """

    if encode_type == 'float32':
        encoder = Float32Encoder(sign_bit=sign_bit, exp_bits=exp_bits, mantissa_bits=mantissa_bits)
        since = time()
        encoded_train_conv_features = encoder.transform(train_conv_features)
        train_encode_time = time() - since

        since = time()
        encoded_test_conv_features = encoder.transform(test_conv_features)
        test_encode_time = time() - since

    elif encode_type == 'abs_th':
        since = time()
        encoded_train_conv_features = (np.abs(train_conv_features) >= threshold).view(np.uint8)
        train_encode_time = time() - since

        since = time()
        encoded_test_conv_features = (np.abs(test_conv_features) >= threshold).view(np.uint8)
        test_encode_time = time() - since

    elif encode_type == 'plain_th':
        since = time()
        encoded_train_conv_features = (train_conv_features >= threshold).view(np.uint8)
        train_encode_time = time() - since

        since = time()
        encoded_test_conv_features = (test_conv_features >= threshold).view(np.uint8)
        test_encode_time = time() - since

    else:
        print('ERROR: encode type not understood.')
        return
    print("{0} encoding time: train={1:2.4f} s\ttest = {2:2.4f} s\n"
          .format(encode_type, train_encode_time, test_encode_time))

    return train_encode_time, test_encode_time, encoded_train_conv_features, encoded_test_conv_features


def get_random_features(X, n_components):
    """
    Performs the random projection of the encoded random features X using the OPU.

    Parameters
    ----------
    X: numpy 2d array,
        encoded convolutional training features. Make sure that the dtype is int8 if n_components!=0.
    n_components: int,
        number of random projections.

    Returns
    -------

    proj_time: float,
        projection time for the features.
    random_features: numpy 2d array,
        random features of the training set. If n_components=0, the original train random features are returned.

    """

    if n_components == 0:
        train_time = 0.

        # The conversion to numpy is needed for compatibility with the MixingBitPlanDecoder.
        if type(X) is not np.ndarray:
            X = X.numpy()

        return train_time, X

    with OPU(n_components=n_components) as opu:
        since = time()
        random_features = opu.transform1d(X)
        train_time = time() - since

    return train_time, random_features


def decoding(random_features, exp_bits=1, sign_bit=False, mantissa_bits=0, decode_type='mixing'):
    """
    Decodes the random features.

    Parameters
    ----------

    random_features: numpy 2d array,
        random features.
    exp_bits: int,
        number of bits for the exponent. Defaults to 1.
    sign_bit: boolean,
        if True, a bit will be used for the sign of the encoded numbers. Defaults to False.
    mantissa_bits: int,
        number of bits for the mantissa. Defaults to 0.
    encode type: string,
        type of decoding to perform. If not 'mixing', it just converts the dtype to float32 for better compatibility
        with scikit-learn.

    Returns
    -------
    decoding_time: float,
        encoding time.
    dec_train_random_features: numpy 2d array,
        decoded random features.
    """

    if decode_type == 'mixing':
        if sign_bit is True:
            decode_bits = exp_bits + mantissa_bits + 1
        else:
            decode_bits = exp_bits + mantissa_bits
        decoder = MixingBitPlanDecoder(n_bits=decode_bits)
        since = time()
        dec_random_features = decoder.transform(random_features)
        train_decode_time = time() - since

    else:
        since = time()
        dec_random_features = random_features.astype('float32')
        train_decode_time = time() - since


    return train_decode_time, dec_random_features


def dummy_predict_GPU(clf, test_random_features, device='cpu', dtype="float32"):
    """
    Performs a dummy decoding + predict on the random features on CPU or GPU.

    Parameters
    ----------
    clf: Ridge classifier,
        Ridge classifier object from scikit-learn library.
    test_random_features: numpy array,
        random features of the test set.
    device: string,
        device to use for the computation. Choose between 'cpu' and 'gpu:x', where
        x is GPU number. Defaults to 'cpu'.

    Returns
    -------
    predict_time: float,
        prediction time.
    """
    if dtype == "float32":
        tensor_dtype = torch.float
    elif dtype == "float16":
        tensor_dtype = torch.half

    x = torch.ByteTensor(test_random_features).to(device)

    torch.cuda.synchronize()
    start = time()
    x = x.type(tensor_dtype)
    torch.cuda.synchronize()
    decode_time = time() - start


    ridge_coefficients = torch.FloatTensor(clf.coef_.T).to(device).type(tensor_dtype)
    start = time()
    torch.cuda.synchronize()
    torch.mm(x, ridge_coefficients)
    torch.cuda.synchronize()
    predict_time = time() - start

    del ridge_coefficients, x
    torch.cuda.empty_cache()

    return decode_time, predict_time

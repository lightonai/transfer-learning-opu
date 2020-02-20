import os
from time import time

import torch
import numpy as np

import tensorrt as trt

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import ctypes



def trt_conv_features(loader, engine, out_shape):
    """
    Computes the convolutional features of the images in the loader using the TensorRT library.

    Parameters
    ----------
    loader: Pytorch dataloader,
        loader containg the image/label pairs.
    engine: TensorRT engine,
        engine build from the .onnx file of a model.
    out_shape: int,
        flattened shape of the convolutional features associated to the images. For example, the full vgg16 has 25088
        features per image.

    Returns
    -------
    conv_features: numpy array,
        array containing the convolutional features. format is (# of samples * # of features).
        They are already moved to CPU.
    labels: list of int,
        labels associated to each image.
    full_conv_time: float,
        time required to compute the convolutional features. It includes the data loading.
    conv_only_time: float,
        time required to compute the convolutional features without the dataloading.

    """
    conv_only_time = 0

    n_images = len(loader.dataset)
    batch_size = loader.batch_size

    conv_features = np.zeros((n_images, out_shape), dtype="float32")
    labels = np.empty(n_images, dtype='uint8')
    last_batch_buffer = torch.zeros((batch_size, 3, 224, 224))

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    with engine.create_execution_context() as context:

        convolution_start = time()
        for batch_id, (image, target) in enumerate(loader):
            current_batch_size = image.shape[0]

            if current_batch_size != engine.max_batch_size:
                last_batch_buffer[0:image.shape[0]] = image
                image = last_batch_buffer

            np.copyto(inputs.host, image.view(-1))

            conv_no_data_loading_start = time()
            outs = do_inference(context, bindings, inputs, outputs, stream, batch_size=batch_size).reshape(batch_size,
                                                                                                           -1)
            conv_only_time += time() - conv_no_data_loading_start

            conv_features[batch_id * batch_size: (batch_id + 1) * batch_size, :] = outs[:current_batch_size]
            labels[batch_id * batch_size: (batch_id + 1) * batch_size] = target.numpy()

    full_conv_time = time() - convolution_start

    return conv_features, labels, full_conv_time, conv_only_time



def build_engine_onnx(model_file, max_batch_size, max_workspace_size=1, dtype='float32', calibrator=None):
    """
    Builds a GPU engine for the given .onnx model optimized for a given batch size and GPU model.


    Parameters
    ----------

    model_file: string,
        path to the onnx model.
    max_batch_size: int,
        maximum batch size for the images. All the computations will be optimized for this value.
        Lower batch sizes will work, but performance will not be optimal.

    max_workspace_size: int,
        maximum memory reserved for inference in GiB. Defaults to 1.
    dtype: str,
        dtype for the computations. Choose between 'float32', 'float16' and 'int8'. Defaults to 'float32'
        NOTE: when picking lower precision dtypes, make sure that the GPU in use supports computations
            with those dtypes. If that is not the case, TensorRT should return a warning.
    calibrator: tensorrt calibrator object or None,
        calibrator to map the float32 range into the int8 one. Required only if dtype=='int8'. Defaults to None.

    Returns
    -------
    builder.build_cuda_engine(network): tensorrt engine,
        engine optimized for the given batch size and the current GPU.

    """

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = max_workspace_size * 1 << 30
        builder.max_batch_size = max_batch_size

        # Load the Onnx model and parse it in order to populate the TensorRT network.

        if dtype == 'f16':
            builder.fp16_mode = True
            builder.strict_type_constraints = True

        elif dtype == 'int8':
            builder.int8_mode = True
            builder.strict_type_constraints = True
            builder.int8_calibrator = calibrator

        with open(model_file, 'rb') as model:
            parser.parse(model.read())

        return builder.build_cuda_engine(network)


class calibrator(trt.IInt8EntropyCalibrator2):
    """
    Class for the int8 calibrator.
    """

    def __init__(self, loader, cache_file):
        """
        Initializator for the calibrator.

        Parameters
        ----------

        loader: pytorch dataloader,
            loader for the calibration images.
        cache_file: str,
            cache for the calibration result. This allows the  calibrator to load previous calibration results and skip
            the calibration step. File extension is .cache.

        TODO: pass more images in the calibration.
        """

        trt.IInt8EntropyCalibrator2.__init__(self)

        self.shape = loader.dataset[0][0].shape
        self.loader = loader
        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(loader.batch_size * trt.volume(self.shape) * trt.float32.itemsize)

        def load_batches():
            return iter(loader)

        self.batches = load_batches()

    def get_batch_size(self):
        """ Returns the batch size"""
        return self.loader.batch_size

    def get_batch(self, names):
        """
        Performs the calibration.
        This works by taking the images, converting them to bytes and copying them to GPU.
        The code then computes the scaling factors for the float32 -> int8 quantization and returns either a list
        containing pointers to input device buffers, or None, which signals to TensorRT that the batches are over.

        Parameters
        ----------
        names: tensorrt bindings,
            names for the engine bindings.

        """
        try:
            print("Calibrating on a batch...")
            data, target = next(self.batches)
            print("data shape {}\n data target = {}".format(data.shape, target))
            cuda.memcpy_htod(self.device_input, data.detach().numpy().tobytes())
            return [int(self.device_input)]

        except StopIteration:
            return None

    def read_calibration_cache(self):
        """Reads the calibration cache provided when the class was created."""
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        """Writes a calibration cache in the location provided when the class was created."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """
    Allocates the necessary memory to perform inference.

    Parameters
    ----------
    engine: tensorrt engine,
        engine for the neural network of choice.

    Returns
    -------

    inputs: memory allocation address for the input
    outputs: memory allocation address for the output
    bindings: memory bindings
    stream: cuda stream

    """

    stream = cuda.Stream()

    in_size = trt.volume(engine.get_binding_shape(0)) * engine.max_batch_size
    in_dtype = trt.nptype(engine.get_binding_dtype(0))

    out_size = trt.volume(engine.get_binding_shape(1)) * engine.max_batch_size
    out_dtype = trt.nptype(engine.get_binding_dtype(1))

    in_host_mem = cuda.pagelocked_empty(in_size, in_dtype)
    out_host_mem = cuda.pagelocked_empty(out_size, out_dtype)

    in_device_mem = cuda.mem_alloc(in_host_mem.nbytes)
    out_device_mem = cuda.mem_alloc(out_host_mem.nbytes)

    bindings = [int(in_device_mem), int(out_device_mem)]

    inputs = HostDeviceMem(in_host_mem, in_device_mem)
    outputs = HostDeviceMem(out_host_mem, out_device_mem)

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """
    Performs inference on a given batch in a tensorrt context.
    The ideal workflow should be this:

    ====================================================================================
    with build_engine_onnx(onnx_path, max_batch_size, dtype, calibrator=calib) as engine:   <-- build engine
        inputs, outputs, bindings, stream = allocate_buffers(engine)                        <-- allocate memory
        with engine.create_execution_context() as context:                                  <-- open a context
            np.copyto(inputs.host, image.view(-1))                                          <-- flatten the image, then copy it to the allocated memory
            outs = do_inference(context, bindings, inputs, outputs, stream, batch_size=batch_size) <-- inference
    ====================================================================================

    Parameters
    ----------

    context: tensorrt context,
    bindings: memory bindings
    inputs: memory allocation address for the input
    outputs: memory allocation address for the output
    stream: cuda stream
    batch_size: int,
        batch size to use for the computations. Defults to 1.
        NOTE: If using int8 dtype, the batch size CAN be different than the calibration one.

    Returns
    -------
    outputs.host: numpy array,
        Result of the computation. Note that it is a flattened array, so it should be reshaped to (32,-1) or whatever
        is the desired shape.

    """

    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(inputs.device, inputs.host, stream)

    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(outputs.host, outputs.device, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return outputs.host

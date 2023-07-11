import numpy as np
import tensorrt as trt

import time
import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import os
import tensorrt
import common
import ctypes

import onnx_graphsurgeon as gs
import onnx


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt_runtime = trt.Runtime(TRT_LOGGER)
onnx_path = "./checkpoints/patch3.onnx"
engine3_path = "./checkpoints/patch3_engine.trt"
engine2_path = "./checkpoints/patch2_engine.trt"
engine1_path = "./checkpoints/patch1_engine.trt"
engine_model_path = "./checkpoints/model_engine.trt"
engine_repro_path = "./para/reproject.trt"
engine_model_nq_path = "./checkpoints/model_nq_engine.trt"
engine_feature_path = "./checkpoints/feature_engine.trt"
soFile = "./AddScalarPlugin.so"

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.set_flag(trt.BuilderFlag.FP16)
            config.max_workspace_size = common.GiB(1)
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                mode = model.read()
                if not parser.parse(mode):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            for i in range(network.num_inputs):
                tensor = network.get_input(i)
                formats = 1 << int(tensorrt.TensorFormat.LINEAR)
                network.get_input(i).allowed_formats = formats
                network.get_input(i).dtype = tensorrt.DataType.HALF
                if(tensor.name == 'src_proj' or tensor.name == 'ref_proj'):
                    network.get_input(i).dtype = tensorrt.DataType.FLOAT
                print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)
            for i in range(network.num_layers):
                print(network[i].name)
                if(network[i].name == 'Reproject_0'):
                    network[i].precision = tensorrt.DataType.FLOAT
                    if(network[i].precision_is_set == True):
                        print("set True")
            for i in range(network.num_outputs):
                tensor = network.get_output(i)
                network.get_output(i).dtype = tensorrt.DataType.HALF
                print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)
            # network.get_input(0).shape = [1, 64, 150, 200]

            # profile = builder.create_optimization_profile()
            # profile.set_shape('intr_mat', (3, 3), (3, 3), (3, 3))
            # profile.set_shape('src_feature', (1, 64, 150, 200), (1, 64, 150, 200), (1, 64, 150, 200))
            # profile.set_shape('intr_mat_inv', (3, 3), (3, 3), (3, 3))
            # profile.set_shape('src_proj', (1, 4, 4), (1, 4, 4), (1, 4, 4))
            # profile.set_shape('ref_proj', (1, 4, 4), (1, 4, 4), (1, 4, 4))
            # profile.set_shape('depth_sample', (1, 48, 150, 200), (1, 48, 150, 200), (1, 48, 150, 200))
            # profile.set_shape('cdhw', (4,), (4,), (4,))
            # config.add_optimization_profile(profile)

            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()



def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def allocate_buffers(engine, batch_size, data_type):
    """
    This is the function to allocate buffers for input and output in the device
    Args:
       engine : The path to the TensorRT engine.
       batch_size : The batch size for execution time.
       data_type: The type of the data for input and output, for example trt.float32.

    Output:
       h_input_1: Input in the host.
       d_input_1: Input in the device.
       h_output_1: Output in the host.
       d_output_1: Output in the device.
       stream: CUDA stream.

    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream


def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed)


def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width):
    """
    This is the function to run the inference
    Args:
       engine : Path to the TensorRT engine
       pics_1 : Input images to the model.
       h_input_1: Input in the host
       d_input_1: Input in the device
       h_output_1: Output in the host
       d_output_1: Output in the device
       stream: CUDA stream
       batch_size : Batch size for execution time
       height: Height of the output image
       width: Width of the output image

    Output:
       The list of output images

    """

    load_images_to_buffer(pics_1, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.

        context.profiler = trt.Profiler()
        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = h_output.reshape((batch_size, -1, height, width))
        return out


def check_accuracy(context, batch_size):
    inputs, outputs, bindings, stream = common.allocate_buffers(context.engine)

    inputs[0].host = np.random.random((1,321,150,200)).astype(np.float32)
    output = common.do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size
        )
    return output


ctypes.cdll.LoadLibrary(soFile)

# graph = gs.import_onnx(onnx.load("./checkpoints/folded.onnx"))
# graph.cleanup()
# onnx.save(gs.export_onnx(graph), "./checkpoints/removed.onnx")
# engine = get_engine("./checkpoints/removed.onnx", engine_model_path)

# context = engine.create_execution_context()
# #
# output = check_accuracy(context,1)
#
# #print(output)

#engine = get_engine("./checkpoints/reproject.onnx", engine_repro_path)
print("main end-----------")
# # model = onnx.load(onnx_path)
# # model_simp, check = simplify(model)
# # assert check, "Simplified ONNX model could not be validated"
# # onnx.save(model,"./checkpoints/patch3_smp.onnx")


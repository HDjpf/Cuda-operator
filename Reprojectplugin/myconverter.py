from torch2trt import tensorrt_converter, add_missing_trt_tensors
import torch
import tensorrt as trt
import numpy as np
import ctypes, os
import models
soFile = "./AddScalarPlugin.so"
printso = './printcxx.so'

def getAddScalarPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "AddScalar":
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(0), trt.PluginFieldType.FLOAT32))
            print(c.create_plugin(c.name, trt.PluginFieldCollection(parameterList)))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def getprintPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "printcxx":
            parameterList = []
            # parameterList.append(trt.PluginField("scalar", np.float32(0), trt.PluginFieldType.FLOAT32))
            print(c.create_plugin(c.name, trt.PluginFieldCollection(parameterList)))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

@tensorrt_converter('models.patchmatch.Evaluation.differentiable_warping')
def convert_differentiable_warping(ctx):
    input_0 = ctx.method_args[1]
    input_1 = ctx.method_args[2]
    input_2 = ctx.method_args[3]
    input_3 = ctx.method_args[4]
    input_4 = ctx.method_args[5]
    input_5 = ctx.method_args[6]
    input_6 = ctx.method_args[7]

    input_trt_0, input_trt_1, input_trt_2, input_trt_3, input_trt_4, input_trt_5, input_trt_6 = \
        add_missing_trt_tensors(ctx.network, [input_0, input_1, input_2, input_3, input_4, input_5, input_6])

    output = ctx.method_return
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)

    pluginLayer = ctx.network.add_plugin_v2(inputs=[input_trt_0, \
        input_trt_1, input_trt_2, input_trt_3, input_trt_4, input_trt_5, input_trt_6 ], plugin=getAddScalarPlugin())

    # pluginLayer.get_output(0).name = "outputs"
    output._trt = pluginLayer.get_output(0)


@tensorrt_converter('models.patchmatch.printcxx')
def convert_printcxx(ctx):
    input_0 = ctx.method_args[0]
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(printso)

    input_trt_0 = add_missing_trt_tensors(ctx.network,[ input_0])
    pluginLayer = ctx.network.add_plugin_v2(inputs=input_trt_0, plugin=getprintPlugin())

@tensorrt_converter('models.net.printcxx')
def convert_printcxx(ctx):
    input_0 = ctx.method_args[0]
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(printso)

    input_trt_0 = add_missing_trt_tensors(ctx.network,[ input_0])
    pluginLayer = ctx.network.add_plugin_v2(inputs=input_trt_0, plugin=getprintPlugin())

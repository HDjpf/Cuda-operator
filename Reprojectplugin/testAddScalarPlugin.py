#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ctypes
from cuda import cudart
import numpy as np
import os
import tensorrt as trt
from calibrator import Calib
import torch

soFile = "./AddScalarPlugin.so"
np.random.seed(31193)
flag = "fp32"

def printArrayInfomation(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

def addScalarCPU(inputH, scalar):
    return [inputH[0] + scalar]

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "Reproject":
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            print(c.create_plugin(c.name, trt.PluginFieldCollection(parameterList)))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shape, scalar):
    testCase = "<shape=%s,scalar=%f>" % (shape, scalar)
    trtFile = "./model-Dim%s.plan" % str(len(shape))
    print("Test %s" % testCase)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)

    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if flag == "fp16": config.set_flag(trt.BuilderFlag.FP16)
        elif flag == "int8": 
            config.set_flag(trt.BuilderFlag.INT8)


        # shape_c = network.add_constant((3, ), np.array((1, 3, 3), dtype=np.int32))
        # shuffle_1 = network.add_shuffle(fc1.get_output(0))
        # shuffle_1.name = "shuffle1"
        # shuffle_1.set_input(1, shape_c.get_output(0))

        inputT0 = network.add_input("inputT0", trt.float32, (-1, 3, 3))
        inputT1 = network.add_input("inputT1", trt.float32, (-1, -1, -1, -1))
        inputT2 = network.add_input("inputT2", trt.float32, (-1, 3, 3))

        inputT3 = network.add_input("inputT3", trt.float32, (-1, 4, 4))
        inputT4 = network.add_input("inputT4", trt.float32, (-1, 4, 4))
        inputT5 = network.add_input("inputT5", trt.float32, (-1, -1, -1, -1))
        inputT6 = network.add_input("inputT6", trt.float32, (-1, 4))

        profile.set_shape(inputT0.name, (1, 3, 3), (1, 3, 3), (1, 3, 3))
        profile.set_shape(inputT1.name, (1, 1, 150, 200), (1, 64, 150, 200), (1, 64, 600, 800))
        profile.set_shape(inputT2.name, (1, 3, 3), (1, 3, 3), (1, 3, 3))

        profile.set_shape(inputT3.name, (1, 4, 4), (1, 4, 4), (1, 4, 4))
        profile.set_shape(inputT4.name, (1, 4, 4), (1, 4, 4), (1, 4, 4))
        profile.set_shape(inputT5.name, (1, 1, 150, 200), (1, 48, 150, 200), (1, 48, 600, 800))
        profile.set_shape(inputT6.name, (1, 4), (1, 4), (1, 4))

        inputT0_c = network.add_constant(trt.Dims([1, 3, 3]), np.ones([1, 3, 3], dtype=np.float32))
        inputT1_c = network.add_constant(trt.Dims([1, 64, 150, 200]), np.ones([1, 64, 150, 200], dtype=np.float32))
        inputT2_c = network.add_constant(trt.Dims([1, 3, 3]), np.ones([1, 3, 3], dtype=np.float32))
        inputT3_c = network.add_constant(trt.Dims([1, 4, 4]), np.ones([1, 4, 4], dtype=np.float32))
        inputT4_c = network.add_constant(trt.Dims([1, 4, 4]), np.ones([1, 4, 4], dtype=np.float32))
        inputT5_c = network.add_constant(trt.Dims([1, 48, 150, 200]), np.ones([1, 48, 150, 200], dtype=np.float32))
        inputT6_c = network.add_constant(trt.Dims([1, 4]), np.ones([1, 4], dtype=np.float32))



        intr_mat = torch.load("./para/intr_mat.pt").cpu().numpy()
        src_feature = torch.load("./para/src_feature.pt").cpu().numpy()
        intr_mat_inv = torch.load("./para/intr_mat_inv.pt").cpu().numpy()
        src_proj = torch.load("./para/src_proj.pt").cpu().numpy()
        ref_proj = torch.load("./para/ref_proj.pt").cpu().numpy()
        depth_sample = torch.load("./para/depth_sample.pt").cpu().numpy()
        cdhw = torch.load("./para/cdhw.pt").cpu().numpy()
        warped_feature = torch.load("./para/warped_feature.pt").cpu().numpy()

        config.add_optimization_profile(profile)
        # pluginLayer = network.add_plugin_v2(inputs=[inputT0, inputT1, inputT2, inputT3, inputT4,inputT5,inputT6], plugin=getAddScalarPlugin(scalar))
        pluginLayer = network.add_plugin_v2(inputs=[
            inputT0_c.get_output(0),inputT1_c.get_output(0), inputT2_c.get_output(0), inputT3_c.get_output(0), inputT4_c.get_output(0),inputT5_c.get_output(0),inputT6_c.get_output(0)], plugin=getAddScalarPlugin(scalar))

        pluginLayer.get_output(0).name = "outputs"
        network.mark_output(pluginLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)

        print("-------")
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, (1, 3, 3))#[32, 32, 32]
    context.set_binding_shape(1, (1, 64, 150, 200))
    context.set_binding_shape(2, (1, 3, 3))  # [32, 32, 32]
    context.set_binding_shape(3, (1, 4, 4))
    context.set_binding_shape(4, (1, 4, 4))
    context.set_binding_shape(5, (1, 48, 150, 200))
    context.set_binding_shape(6, (1, 4))

    print(context.all_binding_shapes_specified)

    #print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput

    #for i in range(nInput):
    #    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    #for i in range(nInput, nInput + nOutput):
    #    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))


    bufferH = []

    bufferH.append(intr_mat) #intr_mat
    bufferH.append(src_feature)  #src_fea
    bufferH.append(intr_mat_inv) #intr_mat_inv
    bufferH.append(src_proj) #src_proj
    bufferH.append(ref_proj) #ref_proj
    bufferH.append(depth_sample) #depth map
    bufferH.append(cdhw)

    # print(trt.nptype(engine.get_binding_dtype(nInput)))
    for i in range(nOutput):
        if flag == "fp32": bufferH.append(np.zeros(context.get_binding_shape(nInput + i), dtype=np.float32))
        elif flag == "fp16": bufferH.append(np.zeros(context.get_binding_shape(nInput + i), dtype=np.float16))#trt.nptype(engine.get_binding_dtype(nInput + i))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print("-----------output")
    print(bufferH[-1])
    # outputCPU = addScalarCPU(bufferH[:nInput], scalar)
    """
    for i in range(nInput):
        printArrayInfomation(bufferH[i])
    for i in range(nOutput):
        printArrayInfomation(bufferH[nInput+i])
    for i in range(nOutput):
        printArrayInfomation(outputCPU[i])
    """
    # check(bufferH[nInput:][0], outputCPU[0], True)

    for buffer in bufferD:
        cudart.cudaFree(buffer)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run([32], 1)
    # run([32, 32], 1)
    # run([16, 16, 16], 1)
    # run([8, 8, 8, 8], 1)

    print("Test all finish!")

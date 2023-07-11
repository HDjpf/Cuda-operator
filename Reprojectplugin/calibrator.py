import os
import numpy as np
from cuda import cudart
import tensorrt as trt

class Calib(trt.IInt8EntropyCalibrator2):

    def __init__(self, inputs, cachefile) -> None: # input_shapes: [input0, input1, input2, input3, input4] or data.Dataset
        self.total_data = inputs[0][0]

        self.bufferD = []
        for input_ in inputs:
            self.bufferD.append(int(cudart.cudaMalloc(input_[0].nbytes)[1]))

        self.inputs = inputs
        self.count = 0
        self.cachefile = cachefile

    def __del__(self):
        for addr in self.bufferD:
            cudart.cudaFree(addr)

    def get_batch_size(self):   # do NOT change name
        return self.total_data

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        if self.count < self.total_data:

            for input_, addr in zip(self.inputs, self.bufferD):
                data = np.ascontiguousarray(input_[self.count].astype(np.float32))
                cudart.cudaMemcpy(addr, data.ctypes.data, input_[self.count].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            
            self.count += 1
            return self.bufferD
        else:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cachefile):
            print("Succeed finding cahce file: %s" % (self.cachefile))
            with open(self.cachefile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cachefile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
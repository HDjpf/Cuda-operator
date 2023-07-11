import time

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .patchmatch import *

# import torch_tensorrt as torchtrt
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules, tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
# from test_trt import *
# import onnxruntime


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)
        
    
        self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        self.output2 = nn.Conv2d(64, 32, 1, bias=False)
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)

        torch.nn.init.orthogonal(self.output1.weight)
        torch.nn.init.orthogonal(self.inner1.weight)
        torch.nn.init.orthogonal(self.inner2.weight)
        torch.nn.init.orthogonal(self.output2.weight)
        torch.nn.init.orthogonal(self.output3.weight)


    def feature(self,x):
        output_feature = {}

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        output_feature3 = self.output1(conv10)

        intra_feat = F.interpolate(conv10, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner1(conv7)
        del conv7, conv10
        output_feature2 = self.output2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(
            conv4)
        del conv4
        output_feature1 = self.output3(intra_feat)

        del intra_feat

        # output_feature3 = output_feature3.repeat(1, 1, 4, 4)
        # output_feature2 = output_feature2.repeat(1, 1, 2, 2)
        # output_feature = torch.cat([output_feature3, output_feature2, output_feature1], 1)
        return output_feature3, output_feature2, output_feature1

    def forward(self, imgs_0):

        features3 = []
        features2 = []
        features1 = []
        for img in imgs_0:
            output_feature3, output_feature2, output_feature1 = self.feature(img)
            features1.append(output_feature1)
            features2.append(output_feature2)
            features3.append(output_feature3)
        return features3, features2, features1

        # output_feature={}
        #
        # conv1 = self.conv1(self.conv0(x))
        # conv4 = self.conv4(self.conv3(self.conv2(conv1)))
        #
        # conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        # conv10 = self.conv10(self.conv9(self.conv8(conv7)))
        #
        # output_feature3 = self.output1(conv10)
        #
        # intra_feat = F.interpolate(conv10, scale_factor=2.0, mode="bilinear",align_corners=False) + self.inner1(conv7)
        # del conv7, conv10
        # output_feature2 = self.output2(intra_feat)
        #
        # intra_feat = F.interpolate(intra_feat, scale_factor=2.0, mode="bilinear",align_corners=False) + self.inner2(conv4)
        # del conv4
        # output_feature1 = self.output3(intra_feat)
        #
        # del intra_feat
        #
        # # output_feature3 = output_feature3.repeat(1, 1, 4, 4)
        # # output_feature2 = output_feature2.repeat(1, 1, 2, 2)
        # # output_feature = torch.cat([output_feature3, output_feature2, output_feature1], 1)
        # return output_feature3, output_feature2, output_feature1
        

class Refinement(nn.Module):
    def __init__(self):
        
        super(Refinement, self).__init__()
        
        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(3, 8)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(1, 8)
        self.conv2 = ConvBnReLU(8, 8)
        self.deconv = nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        
        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(16, 8)
        self.res = nn.Conv2d(8, 1, 3, padding=1, bias=False)
        torch.nn.init.orthogonal(self.deconv.weight)
        torch.nn.init.orthogonal(self.res.weight)
        
        
    def forward(self, img, depth_0, depth_min, depth_max):
        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0-depth_min.view(batch_size,1,1,1))/(depth_max.view(batch_size,1,1,1)-depth_min.view(batch_size,1,1,1))
        
        conv0 = self.conv0(img)
        conv_res = self.conv2(self.conv1(depth))

        deconv_res = self.deconv(conv_res)

        deconv = F.relu(self.bn(deconv_res), inplace=True)
        
        # printcxx(conv0)
        cat = torch.cat((deconv, conv0), dim=1)

        # depth residual
        res = self.res(self.conv3(cat))

        
        depth = F.interpolate(depth,scale_factor=2, mode="area")

        depth = depth + res
        # convert the normalized depth back
        depth = depth * (depth_max.view(batch_size,1,1,1)-depth_min.view(batch_size,1,1,1)) + depth_min.view(batch_size,1,1,1)

        return depth



class PatchmatchNet(nn.Module):
    def __init__(self, patchmatch_interval_scale = [0.005, 0.0125, 0.025], propagation_range = [6,4,2],
                patchmatch_iteration = [1,2,2], patchmatch_num_sample = [8,8,16], propagate_neighbors = [0,8,16],
                evaluate_neighbors = [9,9,9]):
        super(PatchmatchNet, self).__init__()

        self.stages = 4
        self.feature = FeatureNet()
        #self.feature_ts = torch.jit.load("./checkpoints/trt_ts_feature.ts")
        self.patchmatch_num_sample = patchmatch_num_sample
        
        num_features = [8, 16, 32, 64]
        
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # number of groups for group-wise correlation
        self.G = [4,8,8]

        # self.engine_feature = get_engine("./checkpoints/feature_folded.onnx", engine_feature_path)
        # self.inputs_feature, self.outputs_feature, self.bindings_feature, self.stream_feature = \
        #     common.allocate_buffers(self.engine_feature)
        # self.context_feature = self.engine_feature.create_execution_context()
        #
        # self.engine_p3 = get_engine("./checkpoints/patch3_folded.onnx", engine3_path)
        # self.inputs3, self.outputs3, self.bindings3, self.stream3 = common.allocate_buffers(self.engine_p3)
        # self.context_p3 = self.engine_p3.create_execution_context()
        #
        # self.engine_p2 = get_engine("./checkpoints/patch2_folded.onnx", engine2_path)
        # self.inputs2, self.outputs2, self.bindings2, self.stream2 = common.allocate_buffers(self.engine_p2)
        # self.context_p2 = self.engine_p2.create_execution_context()
        #
        # self.engine_p1 = get_engine("./checkpoints/patch1_folded.onnx", engine1_path)
        # self.inputs1, self.outputs1, self.bindings1, self.stream1 = common.allocate_buffers(self.engine_p1)
        # self.context_p1 = self.engine_p1.create_execution_context()


        for l in range(self.stages-1):
            
            if l == 2:
                patchmatch = PatchMatch(True, propagation_range[l], patchmatch_iteration[l], 
                            patchmatch_num_sample[l], patchmatch_interval_scale[l],
                            num_features[l+1], self.G[l], self.propagate_neighbors[l], l+1,
                            evaluate_neighbors[l])
            else:
                patchmatch = PatchMatch(False, propagation_range[l], patchmatch_iteration[l], 
                            patchmatch_num_sample[l], patchmatch_interval_scale[l], 
                            num_features[l+1], self.G[l], self.propagate_neighbors[l], l+1,
                            evaluate_neighbors[l])
            setattr(self, f'patchmatch_{l+1}', patchmatch)

        self.upsample_net = Refinement()
        #self.upsample_ts = torch.jit.load("./checkpoints/upsample.ts")

    def depth_regression(self,p, depth_values):
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
        depth = torch.sum(p * depth_values, 1)
        depth = depth.unsqueeze(1)
        return depth

    def collect_stats(self,model, sample, num_batches):
        """Feed data to the network and collect statistic"""

        # Enable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()



        model(sample)



        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    def compute_amax(self,model, **kwargs):
        # Load calib result
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(strict=False, **kwargs)



    def forward(self, imgs,proj_matrices, depth_min, depth_max) : #imgs,proj_matrices, depth_min, depth_max
        
        # imgs_0 = torch.unbind(imgs['stage_0'], 1)

        imgs_0 = [imgs['stage_0'][:, i, ...] for i in range(imgs['stage_0'].size(1))]
        imgs_1 = [imgs['stage_1'][:, i, ...] for i in range(imgs['stage_1'].size(1))]
        imgs_2 = [imgs['stage_2'][:, i, ...] for i in range(imgs['stage_2'].size(1))]
        imgs_3 = [imgs['stage_3'][:, i, ...] for i in range(imgs['stage_3'].size(1))]

        # imgs_1 = torch.unbind(imgs['stage_1'], 1)
        # imgs_2 = torch.unbind(imgs['stage_2'], 1)
        # imgs_3 = torch.unbind(imgs['stage_3'], 1)
        # torch.cuda.synchronize(device=0)
        # time0 = time.time()

        del imgs
        
        self.imgs_0_ref = imgs_0[0]
        self.imgs_1_ref = imgs_1[0]
        self.imgs_2_ref = imgs_2[0]
        self.imgs_3_ref = imgs_3[0]
        del imgs_1, imgs_2, imgs_3

        
        self.proj_matrices_0 = [proj_matrices['stage_0'][:, i, ...] for i in range(proj_matrices['stage_0'].size(1))] #torch.unbind(proj_matrices['stage_0'], 1)
        self.proj_matrices_1 = [proj_matrices['stage_1'][:, i, ...] for i in range(proj_matrices['stage_1'].size(1))]
        self.proj_matrices_2 = [proj_matrices['stage_2'][:, i, ...] for i in range(proj_matrices['stage_2'].size(1))]
        self.proj_matrices_3 = [proj_matrices['stage_3'][:, i, ...] for i in range(proj_matrices['stage_3'].size(1))]
        del proj_matrices
        #torch.cuda.synchronize(device=0)
        time1 = time.time()
        assert len(imgs_0) == len(self.proj_matrices_0), "Different number of images and projection matrices"
        
        # step 1. Multi-scale feature extraction
        # features = []
        # for img in imgs_0:
        #     # torch.cuda.synchronize(device=0)
        #     # timef_1 = time.time()
        #     # output_feature_ = self.feature(img)
        #
        #     output_feature3,output_feature2,output_feature1 =self.feature(img)
        #     # torch.cuda.synchronize(device=0)
        #     # timef_2 = time.time()
        #     # print("feature time :{:.4f}".format((timef_2 - timef_1)*1000))
        #     #output_feature_ = self.feature(img)
        #     # output_feature3 = output_feature_[:, 0:64, 0:150, 0:200]
        #     # output_feature2 = output_feature_[:, 64:96, 0:300, 0:400]
        #     # output_feature1 = output_feature_[:, 96:112, 0:600, 0:800]
        #     output_feature = {}
        #     output_feature[3] = output_feature3
        #     output_feature[2] = output_feature2
        #     output_feature[1] = output_feature1
        #     features.append(output_feature)

        # self.inputs_feature[0].host = np.array(imgs_0[0].cuda(), dtype=np.float32)
        # self.inputs_feature[1].host = np.array(imgs_0[1].cuda(), dtype=np.float32)
        # self.inputs_feature[2].host = np.array(imgs_0[2].cuda(), dtype=np.float32)
        # self.inputs_feature[3].host = np.array(imgs_0[3].cuda(), dtype=np.float32)
        # self.inputs_feature[4].host = np.array(imgs_0[4].cuda(), dtype=np.float32)
        #
        # outputs = common.do_inference_v2(self.context_feature, bindings=self.bindings_feature, inputs=self.inputs_feature,
        #                                      outputs=self.outputs_feature,
        #                                      stream=self.stream_feature)

        # features3,features2,features1 = [],[],[]
        # features3.append(torch.tensor(outputs[0], device="cuda").view(1,64,150,200))
        # features2.append(torch.tensor(outputs[1], device="cuda").view(1,32,300,400))
        # features1.append(torch.tensor(outputs[2], device="cuda").view(1,16,600,800))
        # features3.append(torch.tensor(outputs[3], device="cuda").view(1,64,150,200))
        # features2.append(torch.tensor(outputs[4], device="cuda").view(1,32,300,400))
        # features1.append(torch.tensor(outputs[5], device="cuda").view(1,16,600,800))
        # features3.append(torch.tensor(outputs[6], device="cuda").view(1,64,150,200))
        # features2.append(torch.tensor(outputs[7], device="cuda").view(1,32,300,400))
        # features1.append(torch.tensor(outputs[8], device="cuda").view(1,16,600,800))
        # features3.append(torch.tensor(outputs[9], device="cuda").view(1,64,150,200))
        # features2.append(torch.tensor(outputs[10], device="cuda").view(1,32,300,400))
        # features1.append(torch.tensor(outputs[11], device="cuda").view(1,16,600,800))
        # features3.append(torch.tensor(outputs[12], device="cuda").view(1,64,150,200))
        # features2.append(torch.tensor(outputs[13], device="cuda").view(1,32,300,400))
        # features1.append(torch.tensor(outputs[14], device="cuda").view(1,16,600,800))
        features3, features2, features1 = self.feature(imgs_0)  ###jiangpf-------thread pool
        imgs_0 = list(imgs_0)
        # torch.onnx.export(self.feature,imgs_0,"./checkpoints/feature.onnx")
        # features3,features2,features1 = self.feature(imgs_0)
        features = []
        for i in range(len(imgs_0)):
            output_feature3, output_feature2, output_feature1 = features3[i],features2[i],features1[i]

            output_feature = {}
            output_feature[3] = output_feature3
            output_feature[2] = output_feature2
            output_feature[1] = output_feature1
            features.append(output_feature)

        del imgs_0
        ref_feature, src_features = features[0], features[1:]
        
        depth_min = depth_min#.type(torch.float)
        depth_max = depth_max#.type(torch.float)

        torch.cuda.synchronize(device=0)
        time2 = time.time()
        # step 2. Learning-based patchmatch
        depth = []
        depth_patchmatch = {}
        refined_depth = {}

        for l in reversed(range(1, self.stages)):
            src_features_l = [src_fea[l] for src_fea in src_features]
            projs_l = getattr(self, f'proj_matrices_{l}')
            ref_proj, src_projs = projs_l[0], projs_l[1:]

            if l == 3:
                # # quant_desc_input = QuantDescriptor(calib_method='max')
                # # quant_nn.QuantConv2d.set_default_quant_desc_input(tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
                # quant_nn.QuantConv2d.set_default_quant_desc_weight(
                #     tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
                # # quant_nn.QuantConv3d.set_default_quant_desc_input(tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
                # quant_nn.QuantConv3d.set_default_quant_desc_weight(
                #     tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
                # quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(
                #     tensor_quant.QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL)
                # # with torch.no_grad():
                # #     self.collect_stats(self.patchmatch_3, input, num_batches=2)
                # #     self.compute_amax(self.patchmatch_3, method="percentile", percentile=99.99)
                # quant_nn.TensorQuantizer.use_fb_fake_quant = True
                #
                # self.inputs3[0].host = np.array(ref_feature[l].cuda())
                # self.inputs3[1].host = np.array(src_features_l[0].cuda())
                # self.inputs3[2].host = np.array(src_features_l[1].cuda())
                # self.inputs3[3].host = np.array(src_features_l[2].cuda())
                # self.inputs3[4].host = np.array(src_features_l[3].cuda())
                # self.inputs3[5].host = np.array(ref_proj.cuda())
                # self.inputs3[6].host = np.array(src_projs[0].cuda())
                # self.inputs3[7].host = np.array(src_projs[1].cuda())
                # self.inputs3[8].host = np.array(src_projs[2].cuda())
                # self.inputs3[9].host = np.array(src_projs[3].cuda())
                # self.inputs3[10].host = np.array(depth_min.cuda())
                # self.inputs3[11].host = np.array(depth_max.cuda())

                # torch.onnx.export(self.patchmatch_3, (ref_feature[l], src_features_l,
                #                                       ref_proj, src_projs,
                #                                       depth_min, depth_max, depth), "./checkpoints/patch3.onnx")

                # trt_outputs = common.do_inference_v2(self.context_p3,bindings=self.bindings3,inputs=self.inputs3,outputs=self.outputs3,
                #                                      stream=self.stream3)
                # out1 = torch.tensor(trt_outputs[0],device="cuda").view(1,1,150,200)
                # out2 = torch.tensor(trt_outputs[1],device="cuda").view(1,32,150,200)
                # out3 = torch.tensor(trt_outputs[2],device="cuda").view(1,1,150,200)
                # depth = []
                # depth.append(out1)
                # depth.append(out3)
                # score = out2

                depth,score = self.patchmatch_3(ref_feature[l], src_features_l,
                                        ref_proj, src_projs,
                                        depth_min, depth_max, depth=depth)
            elif l == 2:

                # self.inputs2[0].host = np.array(ref_feature[l].cuda())
                # self.inputs2[1].host = np.array(src_features_l[0].cuda())
                # self.inputs2[2].host = np.array(src_features_l[1].cuda())
                # self.inputs2[3].host = np.array(src_features_l[2].cuda())
                # self.inputs2[4].host = np.array(src_features_l[3].cuda())
                # self.inputs2[5].host = np.array(ref_proj.cuda())
                # self.inputs2[6].host = np.array(src_projs[0].cuda())
                # self.inputs2[7].host = np.array(src_projs[1].cuda())
                # self.inputs2[8].host = np.array(src_projs[2].cuda())
                # self.inputs2[9].host = np.array(src_projs[3].cuda())
                # self.inputs2[10].host = np.array(depth_min.cuda())
                # self.inputs2[11].host = np.array(depth_max.cuda())
                # self.inputs2[12].host = np.array(depth.cuda())
                #
                # trt_outputs = common.do_inference_v2(self.context_p2, bindings=self.bindings2, inputs=self.inputs2,
                #                                      outputs=self.outputs2,
                #                                      stream=self.stream2)
                # out1 = torch.tensor(trt_outputs[0]).view(1, 1, 300, 400)
                # out2 = torch.tensor(trt_outputs[1]).view(1, 16, 300, 400)
                # out3 = torch.tensor(trt_outputs[2]).view(1, 1, 300, 400)
                # depth = []
                # depth.append(out1)
                # depth.append(out3)
                # score = out2


                depth,score = self.patchmatch_2(ref_feature[l], src_features_l,
                                        ref_proj, src_projs,
                                        depth_min, depth_max, depth=depth)
            elif l == 1:

                # # quant_desc_input = QuantDescriptor(calib_method='max')
                # # quant_nn.QuantConv2d.set_default_quant_desc_input(tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
                # quant_nn.QuantConv2d.set_default_quant_desc_weight(
                #     tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
                # # quant_nn.QuantConv3d.set_default_quant_desc_input(tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
                # quant_nn.QuantConv3d.set_default_quant_desc_weight(
                #     tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
                # quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(
                #     tensor_quant.QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL)
                # # with torch.no_grad():
                # #     self.collect_stats(self.patchmatch_3, input, num_batches=2)
                # #     self.compute_amax(self.patchmatch_3, method="percentile", percentile=99.99)
                # quant_nn.TensorQuantizer.use_fb_fake_quant = True

                # self.inputs1[0].host = np.array(ref_feature[l].cuda())
                # self.inputs1[1].host = np.array(src_features_l[0].cuda())
                # self.inputs1[2].host = np.array(src_features_l[1].cuda())
                # self.inputs1[3].host = np.array(src_features_l[2].cuda())
                # self.inputs1[4].host = np.array(src_features_l[3].cuda())
                # self.inputs1[5].host = np.array(ref_proj.cuda())
                # self.inputs1[6].host = np.array(src_projs[0].cuda())
                # self.inputs1[7].host = np.array(src_projs[1].cuda())
                # self.inputs1[8].host = np.array(src_projs[2].cuda())
                # self.inputs1[9].host = np.array(src_projs[3].cuda())
                # self.inputs1[10].host = np.array(depth_min.cuda())
                # self.inputs1[11].host = np.array(depth_max.cuda())
                # self.inputs1[12].host = np.array(depth.cuda())
                #
                # trt_outputs = common.do_inference_v2(self.context_p1, bindings=self.bindings1, inputs=self.inputs1,
                #                                      outputs=self.outputs1,
                #                                      stream=self.stream1)
                # depth = []
                # out1 = torch.tensor(trt_outputs[0]).view(1, 8, 600, 800)
                # out2 = torch.tensor(trt_outputs[1]).view(1, 1, 600, 800)
                # depth.append(out2)
                # score = out1

                # torch.onnx.export(self.patchmatch_1, (ref_feature[l], src_features_l,
                #                                       ref_proj, src_projs,
                #                                       depth_min, depth_max, depth), "./checkpoints/patch1.onnx")

                depth, score = self.patchmatch_1(ref_feature[l], src_features_l,
                                                               ref_proj, src_projs,
                                                               depth_min, depth_max, depth=depth)

            depth_patchmatch[f'stage_{l}'] = depth
            
            depth = depth[-1].detach()
            if l > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = F.interpolate(depth,
                                    scale_factor=2, mode='nearest')

        torch.cuda.synchronize(device=0)
        time3 = time.time()
        # step 3. Refinement
        # depth_ = depth.repeat(1, 1, 2, 2)
        # depth_[0, 0, 600, 800] = depth_min[0]
        # depth_[0, 0, 600, 801] = depth_max[0]
        # img_in = torch.cat((self.imgs_0_ref, depth_), 1)
        # depth = self.upsample_ts(img_in)
        depth = torch.tensor(depth,device="cuda")

        depth = self.upsample_net(self.imgs_0_ref, depth, depth_min, depth_max)
        refined_depth['stage_0'] = depth
        

        if self.training:
            return {"refined_depth": refined_depth, 
                        "depth_patchmatch": depth_patchmatch,
                    }
            
        else:
            num_depth = self.patchmatch_num_sample[0]
            print("score",score.shape)
            padinput = F.pad(score.unsqueeze(1), (0, 0, 0, 0, 1, 2))
            score_sum4 = 4 * F.avg_pool3d(padinput, (4, 1, 1), stride=1, padding=0).squeeze(1)
            # [B, 1, H, W]
            depth_index = self.depth_regression(score, depth_values=torch.arange(num_depth, device=score.device, dtype=torch.float)).int()
            depth_index = torch.clamp(depth_index, 0, num_depth-1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index.long())
            photometric_confidence = F.interpolate(photometric_confidence,
                                        scale_factor=2, mode='nearest')
            photometric_confidence = photometric_confidence.squeeze(1)

            #output = torch.cat((refined_depth["stage_0"],photometric_confidence.unsqueeze(1)),dim=0)
           # torch.cuda.synchronize(device=0)
           # time4 = time.time()
            #print("impara:{:.4f} feature:{:.4f}  patch:{:.4f} refine:{:.4f}".format((time1-time0)*1000,(time2-time1)*1000,(time3-time2)*1000,(time4-time3)*1000))
            #return  output
            return {"refined_depth": refined_depth['stage_0'], #[1,1,1200,1600]
                       # "depth_patchmatch": depth_patchmatch, #don't need
                        "photometric_confidence": photometric_confidence,#[1,1200,1600]
                    }
        

def patchmatchnet_loss(depth_patchmatch, refined_depth, depth_gt, mask):

    stage = 4

    loss = 0
    for l in range(1, stage):
        depth_gt_l = depth_gt[f'stage_{l}']
        mask_l = mask[f'stage_{l}'] > 0.5
        depth2 = depth_gt_l[mask_l]

        depth_patchmatch_l = depth_patchmatch[f'stage_{l}']
        for i in range(len(depth_patchmatch_l)):
            depth1 = depth_patchmatch_l[i][mask_l]
            loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')

    l = 0
    depth_refined_l = refined_depth[f'stage_{l}']
    depth_gt_l = depth_gt[f'stage_{l}']
    mask_l = mask[f'stage_{l}'] > 0.5

    depth1 = depth_refined_l[mask_l]
    depth2 = depth_gt_l[mask_l]
    loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')

    return loss
    
def printcxx(input_0):
    pass
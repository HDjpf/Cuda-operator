import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
# import cv2
import numpy as np

import tensorrt as trt
import os




class DepthInitialization(nn.Module):
    def __init__(self, patchmatch_num_sample = 1):
        super(DepthInitialization, self).__init__()
        self.patchmatch_num_sample = patchmatch_num_sample
        self.intial_depth = torch.load("/home/chenyj/trt_Sample/05-Plugin/myplugin/intial_depth.pt")

    
    def forward(self, random_initialization, min_depth, max_depth, height, width, depth_interval_scale, device=torch.device("cuda"), 
                depth=None):
        
        
        batch_size = min_depth.size()[0]
        if random_initialization:
            # first iteration of Patchmatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
            inverse_min_depth = 1.0 / min_depth
            inverse_max_depth = 1.0 / max_depth
            patchmatch_num_sample = 48 
            # [B,Ndepth,H,W]

            depth_sample = self.intial_depth
            # depth_sample = torch.rand((batch_size, patchmatch_num_sample, height, width), device=device) + \
            #                     torch.arange(0, patchmatch_num_sample, 1, device=device, dtype=torch.float).view(1, patchmatch_num_sample, 1, 1)

            depth_sample = inverse_max_depth.view(batch_size,1,1,1) + depth_sample / patchmatch_num_sample * \
                                    (inverse_min_depth.view(batch_size,1,1,1) - inverse_max_depth.view(batch_size,1,1,1))
            
            depth_sample = 1.0 / depth_sample
           
            return depth_sample
            
        else:
            # other Patchmatch, local perturbation is performed based on previous result
            # uniform samples in an inversed depth range
            if self.patchmatch_num_sample == 1:
                return depth
            else:
                inverse_min_depth = 1.0 / min_depth
                inverse_max_depth = 1.0 / max_depth
                
                depth_sample = torch.arange(-self.patchmatch_num_sample//2, self.patchmatch_num_sample//2, 1,
                                    device=device).view(1, self.patchmatch_num_sample, 1, 1).repeat(batch_size,
                                    1, height, width).float()
                inverse_depth_interval = (inverse_min_depth - inverse_max_depth) * depth_interval_scale
                inverse_depth_interval = inverse_depth_interval.view(batch_size,1,1,1)
                
                depth_sample = 1.0 / depth + inverse_depth_interval * depth_sample
                
                depth_clamped = []
                # del depth
                # for k in range(batch_size):
                #     print("K ", k)
                # print(inverse_max_depth.shape, inverse_min_depth.shape)
                depth_clamped.append(torch.clamp(depth_sample, min=inverse_max_depth.item(), max=inverse_min_depth.item()))
                depth_sample = 1.0 / torch.cat(depth_clamped,dim=0)
                # del depth_clamped
                return depth_sample
                

class Propagation(nn.Module):
    def __init__(self, neighbors = 16):
        super(Propagation, self).__init__()
        self.neighbors = neighbors
        self.conv_2d_feature = nn.Conv2d(1,self.neighbors,1,1,0)
        nn.init.constant_(self.conv_2d_feature.weight,1.0)
    
        # for param in self.conv_2d_feature.parameters():
        #     param.requires_grad = False
    
    def forward(self, batch, height, width, depth_sample, grid, depth_min, depth_max, depth_interval_scale):
        # [B,D,H,W]
        num_depth = depth_sample.size()[1]  
        batch = depth_sample.size()[0]    
        propogate_depth = depth_sample.new_empty(batch, num_depth + self.neighbors, height, width)
        
        propogate_depth[:,0:num_depth,:,:] = depth_sample
        
        ###--------jiangpf----------
       
        propogate_depth_sample = self.conv_2d_feature(depth_sample[:,num_depth // 2,:,:].unsqueeze(1))    #---------jiangpf sample 0 define
        ###--------


        
        propogate_depth[:,num_depth:,:,:] = propogate_depth_sample
        
        # sort
        #propogate_depth, _ = torch.sort(propogate_depth, dim=1)

        return propogate_depth
        
        
        

class Evaluation(nn.Module):
    def __init__(self,  G=8, stage=3, evaluate_neighbors=9, iterations=2):
        super(Evaluation, self).__init__()
        
        self.iterations = iterations
        trtFile = "./model_reproject.plan"
        self.G = G
        self.stage = stage
        if self.stage == 3:
            self.pixel_wise_net = PixelwiseNet(self.G)

        if self.stage == 0:
            self.intr_mat = torch.tensor([[2892.33008,0,823.20398],[0,2883.16992,619.07001],[0,0,1]],device="cuda")
            self.intr_mat_inv = torch.tensor([[3.4574e-4, 0, -0.28461620],[0, 3.4684e-4, -0.21471854],[0, 0, 1]],
                                        device="cuda")
        elif self.stage == 1:
            self.intr_mat = torch.tensor([[1446.16504,0,411.60199],[0,1441.58496,309.53500],[0,0,1]],device="cuda")
            self.intr_mat_inv = torch.tensor([[6.9148e-4, 0, -0.28461620],[0, 6.9368e-4, -0.21471854],[0, 0, 1]],
                                        device="cuda")
        elif self.stage == 2:
            self.intr_mat = torch.tensor([[723.08252,0,205.80099],[0,720.79248,154.76750],[0,0,1]],device="cuda")
            self.intr_mat_inv = torch.tensor([[1.38297e-3, 0, -0.28461620],[0, 1.38736e-3, -0.21471854],[0, 0, 1]],
                                        device="cuda")
        elif self.stage == 3:
            self.intr_mat = torch.tensor([[361.54126,0,102.90050],[0,360.39624,77.38375],[0,0,1]],device="cuda")
            self.intr_mat_inv = torch.tensor([[2.76594e-3,0,-0.28461620],[0,2.77472e-3,-0.21471854],[0,0,1]],device="cuda")
        
        self.similarity_net = SimilarityNet(self.G, evaluate_neighbors, self.stage)
        if os.path.isfile(trtFile):
            with open(trtFile, "rb") as f:
                logger = trt.Logger(trt.Logger.WARNING)
                engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            if engine == None:
                print("Failed loading engine!")
                return
            self.context = engine.create_execution_context()

            print("Succeeded loading engine!")
        else:
            print("Failed loading trtfile")
        self.reproject_engine = trt.Runtime

    def bilinear_interpolate_torch(self, im, y, x):
        '''
           im : B,C,H,W
           y : 1,numPoints -- pixel location y float
           x : 1,numPOints -- pixel location y float
        '''
        batch,_,_,_ = im.size()
        x0 = torch.floor(x).type(torch.LongTensor)
        x1 = x0 + 1

        y0 = torch.floor(y).type(torch.LongTensor)
        y1 = y0 + 1

        wa = (x1.type(torch.cuda.FloatTensor) - x) * (y1.type(torch.cuda.FloatTensor) - y)
        wb = (x1.type(torch.cuda.FloatTensor) - x) * (y - y0.type(torch.cuda.FloatTensor))
        wc = (x - x0.type(torch.cuda.FloatTensor)) * (y1.type(torch.cuda.FloatTensor) - y)
        wd = (x - x0.type(torch.cuda.FloatTensor)) * (y - y0.type(torch.cuda.FloatTensor))
        # Instead of clamp
        n1 = x1 / im.shape[3]
        n2 = y1 / im.shape[2]
        x1 = x1 - torch.floor(n1).int()
        y1 = y1 - torch.floor(n2).int()
        Ia = []
        Ib = []
        Ic = []
        Id = []
        for i in range(batch):
            Ia.append(im[i:i+1, :, y0[i], x0[i]])
            Ib.append(im[i:i+1, :, y1[i], x0[i]])
            Ic.append(im[i:i+1, :, y0[i], x1[i]])
            Id.append(im[i:i+1, :, y1[i], x1[i]])
        Ia = torch.cat(Ia, dim=0)
        Ib = torch.cat(Ib, dim=0)
        Ic = torch.cat(Ic, dim=0)
        Id = torch.cat(Id, dim=0)
        wa = wa.unsqueeze(1)
        wb = wb.unsqueeze(1)
        wc = wc.unsqueeze(1)
        wd = wd.unsqueeze(1)
        return Ia * wa + Ib * wb + Ic * wc + Id * wd

    def differentiable_warping(self,intr_mat,src_fea,intr_mat_inv, src_proj, ref_proj, depth_samples,cdhw):
        # src_fea: [B, C, H, W]
        # src_proj: [B, 4, 4]
        # ref_proj: [B, 4, 4]
        # depth_samples: [B, Ndepth, H, W]
        # out: [B, C, Ndepth, H, W]
        batch, channels, height, width = src_fea.shape
        num_depth = depth_samples.shape[1]

        torch.cuda.synchronize(0)
        time1 = time.time()

        with torch.no_grad():
        
            rot_src = src_proj[:, :3, :4]
            rot_ref = ref_proj[:, :3, :4]
            src_proj_ = torch.matmul(intr_mat_inv,rot_src)
            ref_proj_ = torch.matmul(intr_mat_inv,rot_ref)
            proj_ = torch.matmul(src_proj_[:,:3,:3],ref_proj_[:,:3,:3].transpose(1, 2))
            trans =torch.matmul(intr_mat,src_proj_[:,:3,3:4] - torch.matmul(proj_,ref_proj_[:,:3,3:4]))
            rot = torch.matmul(torch.matmul(intr_mat,proj_),intr_mat_inv)

            # proj = torch.matmul(src_proj,
            #                     torch.inverse(ref_proj))
            #
            # rot = proj[:, :3, :3]  # [B,3,3]
            # trans = proj[:, :3, 3:4]  # [B,3,1]

            y = torch.arange(0, height, dtype=torch.float32, device=src_fea.device).unsqueeze(1).repeat(1, width)
            x = torch.arange(0, width, dtype=torch.float32, device=src_fea.device).unsqueeze(0).repeat(height, 1)

            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(int(height * width)), x.view(int(height * width))
            xyz = torch.stack((x, y, torch.ones(x.size(), device="cuda")))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
            # ###----------jiangpf-----------
            depth_samples = depth_samples.view(batch, num_depth, int(height * width))
         
            rot_xyz = torch.matmul(rot, xyz)
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                                  height * width)
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
            mask = proj_xyz[:, 2:] > 1e-3

            proj_xyz *= mask
            mask = ~mask
            mask_xyz = torch.ones(mask.size(), device="cuda") * mask

            mask_z = mask_xyz * 1
 
            proj_xyz[:, 2:3] += mask_z
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
            mask_x_max = proj_xy[:, 0:1] < width
            mask_x_min = proj_xy[:, 0:1] >= 0
            proj_xy[:, 0:1] *= mask_x_max
            proj_xy[:, 0:1] *= mask_x_min
            mask_y_max = proj_xy[:, 1:2] < height
            mask_y_min = proj_xy[:, 1:2] >= 0
            proj_xy[:, 1:2] *= mask_y_max
            proj_xy[:, 1:2] *= mask_y_min
            torch.cuda.synchronize(0)
            time2 = time.time()
            warped_src_fea = self.bilinear_interpolate_torch(src_fea, proj_xy[:, 1, :, :], proj_xy[:, 0, :, :]).squeeze(
                2)
            warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
        #
        #
        # warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
        torch.cuda.synchronize(0)
        time2 = time.time()

        return warped_src_fea
        
    
    def forward(self, ref_feature:torch.Tensor, src_features, ref_proj, src_projs, depth_sample, depth_min, depth_max, iter, grid=None, weight=None):
        torch.cuda.synchronize(0)
        time1_o = time.time()
        num_src_features = len(src_features)
        num_src_projs = len(src_projs)
        batch, feature_channel, height, width = ref_feature.size()
        device = ref_feature.get_device()
        
        num_depth = depth_sample.size()[1]
        assert num_src_features == num_src_projs, "Patchmatch Evaluation: Different number of images and projection matrices"
        #if view_weights != None:
            #assert num_src_features == view_weights.size()[1], "Patchmatch Evaluation: Different number of images and view weights"
        
        pixel_wise_weight_sum = 0
        
        ref_feature = ref_feature.view(batch, self.G, feature_channel//self.G, height, width)

        similarity_sum = 0
        torch.cuda.synchronize(0)
        time2_o = time.time()
        if self.stage == 3:
            torch.cuda.synchronize(0)
            time1 = time.time()
           
            for src_feature, src_proj in zip(src_features, src_projs):

                batch, channels, height, width = src_feature.shape

                batch, num_depth, height, width = depth_sample.shape

                cdhw = torch.tensor([channels,num_depth,height,width], dtype=torch.float).cuda()
                
                
                warped_feature = self.differentiable_warping(self.intr_mat,src_feature,self.intr_mat_inv, src_proj, ref_proj, depth_sample,cdhw)
              
                warped_feature = warped_feature.view(batch, self.G, feature_channel//self.G, num_depth, height, width)
                # group-wise correlation
                # ref_feature = ref_feature.unsqueeze(3)
                torch.cuda.synchronize(0)
                similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
 
                similarity_sum += similarity
                    
                # del warped_feature, src_feature, src_proj, similarity, #view_weight
            torch.cuda.synchronize(0)

            score = self.similarity_net(similarity_sum, grid, weight)
            #del similarity, grid, weight
            
            # apply softmax to get probability
            softmax = nn.LogSoftmax(dim=1)
            score = softmax(score)
            score = torch.exp(score)
            
            # depth regression: expectation
            depth_sample = torch.sum(depth_sample * score, dim = 1)
            torch.cuda.synchronize(0)

            return depth_sample, score
        else:
            i=0
            torch.cuda.synchronize(0)
            time1 = time.time()
            for src_feature, src_proj in zip(src_features, src_projs):
                batch, channels, height, width = src_feature.shape
                num_depth = depth_sample.shape[1]
                cdhw = torch.tensor([channels,num_depth,height,width], dtype=torch.float).cuda()
            
                #(depth_sample)
                warped_feature = self.differentiable_warping(self.intr_mat, src_feature, self.intr_mat_inv, src_proj,
                                                             ref_proj, depth_sample, cdhw)

                warped_feature = warped_feature.view(batch, self.G, feature_channel//self.G, num_depth, height, width)
                similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
                # reuse the pixel-wise view weight from first iteration of Patchmatch on stage 3
                #view_weight = view_weights[:,i].unsqueeze(1) #[B,1,H,W]
                i=i+1

                similarity_sum += similarity
                
                # del warped_feature, src_feature, src_proj, similarity, view_weight
            torch.cuda.synchronize(0)
            time2 = time.time()


            
            score = self.similarity_net(similarity_sum, grid, weight)

            softmax = nn.LogSoftmax(dim=1)
            score = softmax(score)
            score = torch.exp(score)


            if self.stage == 1 and iter == self.iterations: 
                # depth regression: inverse depth regression
                ones = torch.ones([1], device=torch.device("cuda"), dtype=torch.float32)
                depth_index = torch.arange(0, num_depth, 1, device=torch.device("cuda"), dtype=torch.float32).view(1, num_depth, 1, 1)
                depth_index = torch.sum(depth_index * score, dim = 1)
                
                inverse_min_depth = ones / depth_sample[:,-1,:,:]
                inverse_max_depth = ones / depth_sample[:,0,:,:]
                depth_sample = inverse_max_depth + depth_index / torch.tensor(num_depth - 1, dtype=torch.float32).cuda() * \
                                            (inverse_min_depth - inverse_max_depth)
                depth_sample = ones / depth_sample

                torch.cuda.synchronize(0)
                time3 = time.time()
                # print("2defferential warp for time1:{:.4f} time2:{:.4f} time3:{:.4f} ".format((time2_o - time1_o) * 1000,
                #                                                                             (time2 - time1) * 1000,
                #                                                                             (time3 - time2) * 1000))
                return depth_sample, score
            
            # depth regression: expectation
            else:
                depth_sample = torch.sum(depth_sample * score, dim = 1)

                torch.cuda.synchronize(0)
                time3 = time.time()
                # print("3defferential warp for time1:{:.4f} time2:{:.4f} time3:{:.4f} ".format((time2_o - time1_o) * 1000,
                #                                                                             (time2 - time1) * 1000,
                #                                                                             (time3 - time2) * 1000))
                return depth_sample, score
            


class PatchMatch(nn.Module):
    def __init__(self, random_initialization = False, propagation_out_range = 2, 
                patchmatch_iteration = 2, patchmatch_num_sample = 16, patchmatch_interval_scale = 0.025,
                num_feature = 64, G = 8, propagate_neighbors = 16, stage=3, evaluate_neighbors=9):
        super(PatchMatch, self).__init__()
        self.random_initialization = random_initialization
        self.depth_initialization = DepthInitialization(patchmatch_num_sample)
        self.propagation_out_range = propagation_out_range
        self.propagation = Propagation(propagate_neighbors)
        self.patchmatch_iteration = patchmatch_iteration

        self.patchmatch_interval_scale = patchmatch_interval_scale
        self.propa_num_feature = num_feature
        # group wise correlation
        self.G = G

        self.stage = stage
        
        self.dilation = propagation_out_range
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        self.evaluation = Evaluation(self.G, self.stage, self.evaluate_neighbors, self.patchmatch_iteration)
        # adaptive propagation
        if self.propagate_neighbors > 0:
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            if not (self.stage == 1 and self.patchmatch_iteration == 1):
                self.propa_conv = nn.Conv2d(
                                self.propa_num_feature,
                                2 * self.propagate_neighbors,
                                kernel_size=3,
                                stride=1,
                                padding=self.dilation,
                                dilation=self.dilation,
                                bias=True)
                nn.init.xavier_uniform_(self.propa_conv.weight)
                nn.init.constant_(self.propa_conv.bias, 0.)

        # adaptive spatial cost aggregation (adaptive evaluation)
        self.eval_conv = nn.Conv2d(self.propa_num_feature, 2 * self.evaluate_neighbors, kernel_size=3, stride=1, 
                                    padding=self.dilation, dilation=self.dilation, bias=True)
        nn.init.xavier_uniform_(self.eval_conv.weight)
        nn.init.constant_(self.eval_conv.bias, 0.)
        self.feature_weight_net = FeatureWeightNet(num_feature, self.evaluate_neighbors, self.G, stage=self.stage)

        ###---------jiangpf-------------
        if self.stage == 3:
            self.conv_2d_feature = nn.Conv2d(48,48,9,1,4)
            self.conv_2d_feature1 = nn.Conv2d(32,32,9,1,4)
        elif self.stage == 2:
            self.conv_2d_feature = nn.Conv2d(16, 16, 9, 1, 4)
            self.conv_2d_feature1 = nn.Conv2d(16, 16, 9, 1, 4)
        elif self.stage == 1:
            self.conv_2d_feature = nn.Conv2d(8, 8, 9, 1, 4)
            self.conv_2d_feature1 = nn.Conv2d(1, 1, 9, 1, 4)
        nn.init.xavier_uniform_(self.conv_2d_feature.weight)
        nn.init.xavier_uniform_(self.conv_2d_feature1.weight)
        ###---------

    # adaptive spatial cost aggregation
    # weight based on depth difference of sampling points and center pixel
    def depth_weight1(self,depth_sample, depth_min, depth_max,offset, patchmatch_interval_scale, evaluate_neighbors):
        # grid: position of sampling points in adaptive spatial cost aggregation
        neighbors = evaluate_neighbors
        batch, num_depth, height, width = depth_sample.size()
        # normalization
        x = 1.0 / depth_sample
        inverse_depth_min = 1.0 / depth_min
        inverse_depth_max = 1.0 / depth_max
        x = (x - inverse_depth_max.view(batch, 1, 1, 1)) / (inverse_depth_min.view(batch, 1, 1, 1) \
                                                            - inverse_depth_max.view(batch, 1, 1, 1))

        ###-------jiangpf------
        x1_ = self.conv_2d_feature(x)

        x1_ = torch.abs(x1_ - x) / patchmatch_interval_scale
        x1_ = torch.clamp(x1_, min=0, max=4)
        x1_ = (-x1_ + 2) * 2
        output = nn.Sigmoid()
        x1_ = output(x1_)



        return x1_.detach()

    # adaptive spatial cost aggregation
    # weight based on depth difference of sampling points and center pixel
    def depth_weight2(self, depth_sample, depth_min, depth_max, grid, patchmatch_interval_scale,
                          evaluate_neighbors):
        # grid: position of sampling points in adaptive spatial cost aggregation
        neighbors = evaluate_neighbors
        batch, num_depth, height, width = depth_sample.size()
        # normalization
        x = 1.0 / depth_sample       ###-----jiangpf---------if depth-smaple is zero which needed modify
        inverse_depth_min = 1.0 / depth_min
        inverse_depth_max = 1.0 / depth_max
        x = (x - inverse_depth_max.view(batch, 1, 1, 1)) / (inverse_depth_min.view(batch, 1, 1, 1) \
                                                                - inverse_depth_max.view(batch, 1, 1, 1))


        ###-------jiangpf------
        x1_ = self.conv_2d_feature1(x)
        # x1_ = x1_.view(batch, num_depth, height, width)
        x1_ = torch.abs(x1_ - x) / patchmatch_interval_scale
        x1_ = torch.clamp(x1_, min=0, max=4)
        x1_ = (-x1_ + 2) * 2
        output = nn.Sigmoid()
        x1_ = output(x1_)

        return x1_.detach()



    def forward(self, ref_feature, src_features, ref_proj, src_projs, depth_min, depth_max,
                depth = None):
        torch.cuda.synchronize(0)
        time1 = time.time()
        depth_samples = []

        device = torch.device("cuda")#ref_feature.get_device()
        batch, _, height, width = ref_feature.size()

        # the learned additional 2D offsets for adaptive propagation
        # if self.propagate_neighbors > 0:
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            # if not (self.stage == 1 and self.patchmatch_iteration == 1):
        propa_offset = torch.tensor([0], dtype=torch.float, device=torch.device("cuda"))#self.propa_conv(ref_feature)

        eval_offset = torch.tensor([0], dtype=torch.float, device=torch.device("cuda")) #self.eval_conv(ref_feature)


        feature_weight = self.feature_weight_net(ref_feature,eval_offset) #eval_offset

        torch.cuda.synchronize(0)
        time2 = time.time()
        
        # first iteration of Patchmatch
        iter = 1
        if self.random_initialization:
            # first iteration on stage 3, random initialization, no adaptive propagation
            depth_sample = self.depth_initialization(True, depth_min, depth_max, height, width, 
                                    self.patchmatch_interval_scale, device)
            # weights for adaptive spatial cost aggregation in adaptive evaluation
            weight = self.depth_weight1(depth_sample, depth_min, depth_max, eval_offset, self.patchmatch_interval_scale,
                                    self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)  #feature and depth position weight
            # weight = weight / torch.sum(weight, dim=2).unsqueeze(2)  #sum deformable conv
            
            # evaluation, outputs regressed depth map and pixel-wise view weights which will
            # be used for subsequent iterations
            
            depth_sample, score = self.evaluation(ref_feature, src_features, ref_proj, src_projs,
                                        depth_sample, depth_min, depth_max, iter, eval_offset, weight, )
          
            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)
        else:
            # subsequent iterations, local perturbation based on previous result
            depth_sample = self.depth_initialization(False, depth_min, depth_max, 
                                height, width, self.patchmatch_interval_scale, device, depth)

            # adaptive propagation
            if self.propagate_neighbors > 0:
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)
                if not (self.stage == 1 and iter == self.patchmatch_iteration):
                    depth_sample = self.propagation(batch, height, width, depth_sample, propa_offset, depth_min, depth_max,
                                            self.patchmatch_interval_scale)
            # weights for adaptive spatial cost aggregation in adaptive evaluation
            weight = self.depth_weight1(depth_sample, depth_min, depth_max, eval_offset, self.patchmatch_interval_scale,
                                    self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)
            weight = weight / torch.sum(weight, dim=2).unsqueeze(2)
            
            # evaluation, outputs regressed depth map
            depth_sample, score = self.evaluation(ref_feature, src_features, ref_proj, src_projs, 
                                        depth_sample, depth_min, depth_max, iter, eval_offset, weight)
            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)

        # torch.cuda.synchronize(0)
        # time3 = time.time()

        for iter in range(2, self.patchmatch_iteration+1):
        #     torch.cuda.synchronize(0)
        #     time3_1 = time.time()
            # local perturbation based on previous result
            depth_sample1 = self.depth_initialization(False, depth_min, depth_max, height, width, self.patchmatch_interval_scale, device, depth_sample)


            # torch.cuda.synchronize(0)
            # time3_2 = time.time()
            # adaptive propagation
            if (self.propagate_neighbors > 0) and not (self.stage == 1 and iter == self.patchmatch_iteration):
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            # if not (self.stage == 1 and iter == self.patchmatch_iteration):
                
                depth_sample1 = self.propagation(batch, height, width, depth_sample1, propa_offset, depth_min, depth_max,
                                            self.patchmatch_interval_scale)
                
            # weights for adaptive spatial cost aggregation in adaptive evaluation

            weight = self.depth_weight2(depth_sample1, depth_min, depth_max, eval_offset, self.patchmatch_interval_scale,
                                    self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)
            # weight = weight / torch.sum(weight, dim=2).unsqueeze(2)
            # torch.cuda.synchronize(0)
            # time3_3 = time.time()
            # evaluation, outputs regressed depth map
            
            depth_sample1, score = self.evaluation(ref_feature, src_features,
                                                ref_proj, src_projs, depth_sample1, depth_min, depth_max, iter, eval_offset, weight)


            depth_sample1 = depth_sample1.unsqueeze(1)
            depth_samples.append(depth_sample1)

        # torch.cuda.synchronize(0)
        # time4 = time.time()
        #print("patch time1:{:.4f} time2:{:.4f} time3:{:.4f} all:{:.4f}".format((time2-time1)*1000, (time3-time2)*1000,(time4-time3)*1000,(time4-time1)*1000))
        return depth_samples, score
        

# first, do convolution on aggregated cost among all the source views
# second, perform adaptive spatial cost aggregation to get final cost
class SimilarityNet(nn.Module):
    def __init__(self, G, neighbors = 9, stage = 3):
        super(SimilarityNet, self).__init__()
        self.neighbors = neighbors
        
        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

        self.stage = stage
        ###--------jiangpf----------
        if self.stage == 3:
            self.conv_2d_feature = ConvBnReLU(48,48,9,1,4)
            self.conv_2d_feature1 = ConvBnReLU(32, 32, 9, 1, 4)
        if self.stage == 2:
            self.conv_2d_feature = ConvBnReLU(16, 16, 9, 1, 4)
        if self.stage == 1:
            self.conv_2d_feature = ConvBnReLU(8, 8, 9, 1, 4)
        nn.init.xavier_uniform_(self.similarity.weight)
        ###---------
        
    def forward(self, x1, grid, weight):
        # x1: [B, G, Ndepth, H, W], aggregated cost among all the source views with pixel-wise view weight
        # grid: position of sampling points in adaptive spatial cost aggregation
        # weight: weight of sampling points in adaptive spatial cost aggregation, combination of 
        # feature weight and depth weight
        
        batch,G,num_depth,height,width = x1.size() 
        
        x1 = self.similarity(self.conv1(self.conv0(x1))).squeeze(1)

        ###-------jiangpf-------
        if self.stage == 3:
            if num_depth == 48:
                x1_ = self.conv_2d_feature(x1)
            elif num_depth == 32:
                print("x1 size",x1.size())
                x1_ = self.conv_2d_feature1(x1)
        elif self.stage == 2:
            if num_depth == 16:
                x1_ = self.conv_2d_feature(x1)
        elif self.stage == 1:
            if num_depth == 8:
                x1_ = self.conv_2d_feature(x1)

    
        return x1_*weight # torch.sum(x1*weight, dim=2)

# adaptive spatial cost aggregation
# weight based on similarity of features of sampling points and center pixel
class FeatureWeightNet(nn.Module):
    def __init__(self, num_feature, neighbors=9, G=8, stage=3):
        super(FeatureWeightNet, self).__init__()
        self.neighbors = neighbors
        self.G = G
        self.stage = stage

        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

        ###----------jiangpf-------------------
        self.conv0_pf = ConvBnReLU(G,16,1,1,0)
        self.conv1_pf = ConvBnReLU(16,8,1,1,0)
        self.similarity_pf = nn.Conv2d(8,1,1,1,0)

        if self.stage == 3:
            self.conv_2d_feature = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=9,stride=1,padding=4)
        elif self.stage == 2:
            self.conv_2d_feature = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=4)
        elif self.stage == 1:
            self.conv_2d_feature = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=9, stride=1, padding=4)
        nn.init.xavier_uniform_(self.similarity.weight)
        nn.init.xavier_uniform_(self.similarity_pf.weight)
        nn.init.xavier_uniform_(self.conv_2d_feature.weight)
        ###--------------------
        
        self.output = nn.Sigmoid()

    def forward(self, ref_feature, offset): #grid,
        # ref_feature: reference feature map
        # grid: position of sampling points in adaptive spatial cost aggregation
        batch,feature_channel,height,width = ref_feature.size()
        
        ###--------jiangpf------------
        ref_feature = ref_feature.view(batch, feature_channel, height, width)
        x_ = self.conv_2d_feature(ref_feature)
        ref_feature_ = ref_feature.view(batch, self.G, feature_channel // self.G, height, width)
        x_ = x_.view(batch, self.G, feature_channel // self.G, height, width)
        x_ = x_ * ref_feature_
        x_ = torch.mean(x_, 2)
        x_ = self.similarity_pf(self.conv1_pf(self.conv0_pf(x_))).squeeze(1)


        return self.output(x_)


# estimate pixel-wise view weight
class PixelwiseNet(nn.Module):
    def __init__(self, G):
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.conv2 = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()
        nn.init.xavier_uniform_(self.conv2.weight)
        

    def forward(self, x1):
        # x1: [B, G, Ndepth, H, W]
        
        # [B, Ndepth, H, W]
        x1 =self.conv2(self.conv1(self.conv0(x1))).squeeze(1)
        
        output = self.output(x1)
        # [B,H,W]
        output = torch.max(output, dim=1)[0]
        
        return output.unsqueeze(1)
        
def printcxx(input_0):
    pass

import torch
import numpy as np
from torch.onnx.symbolic_helper import parse_args
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from test_trt import *

class Reproject(Function):
    @staticmethod
    def forward(ctx,intr_mat,src_fea,intr_mat_inv, src_proj, ref_proj, depth_samples,cdhw):  # ctx 必须要
        # src_fea: [B, C, H, W]
        # src_proj: [B, 4, 4]
        # ref_proj: [B, 4, 4]
        # depth_samples: [B, Ndepth, H, W]
        # out: [B, C, Ndepth, H, W]
        batch, channels, height, width = src_fea.shape
        num_depth = depth_samples.shape[1]


        with torch.no_grad():

            rot_src = src_proj[:, :3, :4]
            rot_ref = ref_proj[:, :3, :4]
            src_proj_ = torch.matmul(intr_mat_inv, rot_src)
            ref_proj_ = torch.matmul(intr_mat_inv, rot_ref)
            proj_ = torch.matmul(src_proj_[:, :3, :3], ref_proj_[:, :3, :3].transpose(1, 2))
            trans = torch.matmul(intr_mat, src_proj_[:, :3, 3:4] - torch.matmul(proj_, ref_proj_[:, :3, 3:4]))
            rot = torch.matmul(torch.matmul(intr_mat, proj_), intr_mat_inv)


            y = torch.arange(0, height, dtype=torch.float32, device=src_fea.device).unsqueeze(1).repeat(1, width)
            x = torch.arange(0, width, dtype=torch.float32, device=src_fea.device).unsqueeze(0).repeat(height, 1)

            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(int(height * width)), x.view(int(height * width))
            xyz = torch.stack((x, y, torch.ones(x.size(), device="cuda")))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]

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

            warped_src_fea = bilinear_interpolate_torch(src_fea, proj_xy[:, 1, :, :], proj_xy[:, 0, :, :])
            warped_src_fea = warped_src_fea.squeeze(2)
            # warped_src_fea = F.grid_sample(src_fea, proj_xy.view(batch, num_depth * height, width, 2), mode='bilinear',
            #                                padding_mode='zeros', align_corners=True)
            warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

        return warped_src_fea

    @staticmethod
    @parse_args('v','v','v','v','v','v','v')
    def symbolic(g, intr_mat, src_feature, intr_mat_inv, src_proj, ref_proj, depth_sample, cdhw):

        return g.op("custom::Reproject", intr_mat, src_feature, intr_mat_inv, src_proj, ref_proj, depth_sample, cdhw
                    )#.setType(src_feature.type().with_dtype(torch.float32).with_sizes(src_feature_shape))



def bilinear_interpolate_torch(im, y, x):
    '''
       im : B,C,H,W
       y : 1,numPoints -- pixel location y float
       x : 1,numPOints -- pixel location y float
    '''
    y_np = y.cpu().numpy()
    x_np = x.cpu().numpy()

    batch, _, _, _ = im.size()
    x0 = torch.floor(x).type(torch.cuda.LongTensor)
    x1 = x0 + 1

    y0 = torch.floor(y).type(torch.cuda.LongTensor)
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
        Ia.append(im[i:i + 1, :, y0[i], x0[i]])
        Ib.append(im[i:i + 1, :, y1[i], x0[i]])
        Ic.append(im[i:i + 1, :, y0[i], x1[i]])
        Id.append(im[i:i + 1, :, y1[i], x1[i]])
    Ia = torch.cat(Ia, dim=0)
    Ib = torch.cat(Ib, dim=0)
    Ic = torch.cat(Ic, dim=0)
    Id = torch.cat(Id, dim=0)
    wa = wa.unsqueeze(1)
    wb = wb.unsqueeze(1)
    wc = wc.unsqueeze(1)
    wd = wd.unsqueeze(1)
    return Ia * wa + Ib * wb + Ic * wc + Id * wd
reproject_ = Reproject.apply
class test_reproject(nn.Module):
    def __init__(self):
        super(test_reproject, self).__init__()

    def forward(self,intr_mat, src_feature,
                                    intr_mat_inv, src_proj,
                                    ref_proj, depth_sample, cdhw):
        warped_feature = reproject_(intr_mat, src_feature,
                                    intr_mat_inv, src_proj,
                                    ref_proj, depth_sample, cdhw)
        return warped_feature

engine = get_engine("./para/reproject.onnx", engine_repro_path)
inputs, outputs_, bindings, stream = common.allocate_buffers(engine)
context = engine.create_execution_context()

intr_mat = torch.load("./para/intr_mat.pt").contiguous()
src_feature = torch.load("./para/src_feature.pt").contiguous()
intr_mat_inv = torch.load("./para/intr_mat_inv.pt").contiguous()
src_proj = torch.load("./para/src_proj.pt").contiguous()
ref_proj = torch.load("./para/ref_proj.pt").contiguous()
depth_sample = torch.load("./para/depth_sample.pt").contiguous()
cdhw = torch.load("./para/cdhw.pt").contiguous()
warped_feature = torch.load("./para/warped_feature.pt").contiguous()

src_proj0 = src_proj#.clone()
ref_proj0 = ref_proj#.clone()
ref_proj_ = ref_proj[:,:,3:4] / 1000
ref_proj[:,0,3] = ref_proj_[:,0,0]
ref_proj[:,1,3] = ref_proj_[:,1,0]
ref_proj[:,2,3] = ref_proj_[:,2,0]

src_proj_ = src_proj[:,:,3:4] / 1000
src_proj[:,0,3] = src_proj_[:,0,0]
src_proj[:,1,3] = src_proj_[:,1,0]
src_proj[:,2,3] = src_proj_[:,2,0]

# context.set_binding_shape(0, (3, 3))
# context.set_binding_shape(1, (1, 64, 150, 200))
# context.set_binding_shape(2, (3, 3))
# context.set_binding_shape(3, (1, 4, 4))
# context.set_binding_shape(4, (1, 4, 4))
# context.set_binding_shape(5, (1, 48, 150, 200))
# context.set_binding_shape(6, (4,))

inputs[0].host = np.array(intr_mat.cpu(),dtype=np.float16)
inputs[1].host = np.array(src_feature.cpu(),dtype=np.float16)
inputs[2].host = np.array(intr_mat_inv.cpu(),dtype=np.float16)
inputs[3].host = np.array(src_proj.cpu(),dtype=np.float32)
inputs[4].host = np.array(ref_proj.cpu(),dtype=np.float32)
inputs[5].host = np.array(depth_sample.cpu(),dtype=np.float16)
inputs[6].host = np.array(cdhw.cpu(), dtype=np.float16)
trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs,
                                     outputs=outputs_,
                                     stream=stream)
out = torch.tensor(trt_outputs[0],dtype=torch.float16).view(1,16,8,600,800).cuda()

repromodel =  test_reproject()
# while 1:
#     torch.cuda.synchronize(0)
#     time1 = time.time()
warped_feature_ = repromodel(intr_mat,src_feature,intr_mat_inv,src_proj0,ref_proj0,depth_sample,cdhw)
    # torch.cuda.synchronize(0)
    # time2 = time.time()
    # print("time :{:.4f}".format((time2 - time1) * 1000))
# torch.onnx.export(repromodel, (intr_mat,src_feature,intr_mat_inv,
#             src_proj,ref_proj,depth_sample,cdhw),"./para/reproject.onnx",opset_version=14,verbose=False)
np_out = out.cpu().numpy()
np_warp = np.array(warped_feature_.cpu(),dtype=np.float16)#warped_feature_.cpu().numpy()
sum_err = torch.sum(torch.abs(out-warped_feature_))
print("---------------")




import torch 
import torch.nn as nn
import time
from torch2trt import TRTModule

import torch2trt
from myconverter import *
from models.patchmatch import PatchMatch
from models.net import PatchmatchNet
class Tp(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1, x2):
        x_1 = x1.reshape((1, 64, 150, 200))
        x_2 = x2.reshape((1, 64, 150, 200))
        return x_1 * x_2



if __name__ == '__main__':

    net = PatchmatchNet().cuda().eval()
    state_dict = torch.load("./model_000008.ckpt")
    net.load_state_dict(state_dict['model'])  # ['model']
    net.eval()
    # min_shape = [(1, 3, 3), (1, 1, 150, 200), (1, 3, 3), (1, 4, 4), (1, 4, 4), (1, 1, 150, 200), (1, 4)]
    # opt_shape = [(1, 3, 3), (1, 16, 600, 800), (1, 3, 3), (1, 4, 4), (1, 4, 4), (1, 16, 600, 800), (1, 4)]
    # max_shape = [(1, 3, 3), (1, 64, 1200, 1600), (1, 3, 3), (1, 4, 4), (1, 4, 4), (1, 48, 1200, 1600), (1, 4)]
    # model_shape = [(1, 64, 150, 200), [(4, 64, 150, 200)], (1, 4, 4), (4, 4, 4), (1), (1)]

    torch.manual_seed(0)
    x0 = {"stage_0": torch.randn([1, 5, 3, 1200, 1600], dtype=torch.float32, device= torch.device("cuda")), \
            "stage_1": torch.randn([1, 5, 3, 600, 800], dtype=torch.float32, device= torch.device("cuda")), \
            "stage_2": torch.randn([1, 5, 3, 300, 400], dtype=torch.float32, device= torch.device("cuda")), \
            "stage_3": torch.randn([1, 5, 3, 150, 200], dtype=torch.float32, device= torch.device("cuda"))}
    
    x1 = {"stage_0": torch.randn([1, 5, 4, 4], dtype=torch.float32, device= torch.device("cuda")), \
            "stage_1": torch.randn([1, 5, 4, 4], dtype=torch.float32, device= torch.device("cuda")), \
            "stage_2": torch.randn([1, 5, 4, 4], dtype=torch.float32, device= torch.device("cuda")), \
            "stage_3": torch.randn([1, 5, 4, 4], dtype=torch.float32, device= torch.device("cuda"))}

    x2 = torch.tensor([425], dtype=torch.float32, device= torch.device("cuda"))
    x3 = torch.tensor([935], dtype=torch.float32, device= torch.device("cuda"))


    model = Tp()

    # model_trt = torch2trt.torch2trt(net, [x0,x1, x2, x3])

    model_trt = TRTModule()

    model_trt.load_state_dict(torch.load('alexnet_trt.pth'))

    out_torch = net(x0, x1, x2, x3)
    while(1):
        torch.cuda.synchronize(0)
        time0 = time.time()
        #out_torch = net(x0, x1, x2, x3)
        out_trt = model_trt(x0, x1, x2, x3)
        torch.cuda.synchronize(0)
        time1 = time.time()
        print("time :{:.4f}".format((time1-time0)*1000),"out_size",out_trt["refined_depth"].shape)
    print(out_trt.keys())
    print(out_torch.keys())
    print(out_torch["refined_depth"] - out_trt["refined_depth"])
    # torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
    # print(net)

    #y_trt = model_trt(x)
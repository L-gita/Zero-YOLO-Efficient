import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k

			
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


# =============================================================================
# 论文 (4.9)~(4.12) 的 Sobel 边缘损失
#   Gx=[[-1,0,1],[-2,0,2],[-1,0,1]]   Gy=[[-1,-2,-1],[0,0,0],[1,2,1]]
#   Ix=I*Gx, Iy=I*Gy
#   M(I)=sqrt(Ix^2+Iy^2)          (4.10 精确版)
#   M(I)=|Ix|+|Iy|                (4.11 工程近似，默认)
#   L_edge = 1/(3N) * sum_{c in RGB} sum_{i=1..N} |M(I_low^c)_i - M(I_enh^c)_i|
# =============================================================================
class SobelMag(nn.Module):
    """逐通道 Sobel 梯度幅值 M(I)。输入 [b,3,H,W]（0~1），输出同尺寸 [b,3,H,W]。"""

    def __init__(self, approx=True, normalize=False):
        super(SobelMag, self).__init__()
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]])          # (4.9) Gx
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]])        # (4.9) Gy
        if normalize:                               # 默认 False，与论文公式完全一致
            kx, ky = kx / 4.0, ky / 4.0
        # conv2d 权重形状 [out_ch, in_ch/groups, kH, kW]；groups=3 -> [3,1,3,3]
        # repeat(3,1,1,1)：R/G/B 三个通道各用一份相同的 Sobel 核
        self.register_buffer('kx', kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1), persistent=False)
        self.register_buffer('ky', ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1), persistent=False)
        self.approx = approx

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == 3, 'SobelMag 需要 3 通道 RGB 输入，收到 %d 通道' % c
        # padding=1 保证输出 H*W=N 不被缩小；replicate 避免边框补 0 造出假边缘
        xp = F.pad(x, (1, 1, 1, 1), mode='replicate')
        Ix = F.conv2d(xp, self.kx, groups=3)        # (4.9) Ix = I * Gx，逐通道并行
        Iy = F.conv2d(xp, self.ky, groups=3)        # (4.9) Iy = I * Gy
        if self.approx:
            M = Ix.abs() + Iy.abs()                 # (4.11)
        else:
            M = torch.sqrt(Ix * Ix + Iy * Iy + 1e-12)   # (4.10)；1e-12 防 sqrt(0) 梯度 NaN
        return M


class L_edge(nn.Module):
    """论文 (4.12)：强制增强前后梯度幅值场一致（零参考，不需要 GT）。"""

    def __init__(self, approx=True, normalize=False):
        super(L_edge, self).__init__()
        self.sobel = SobelMag(approx=approx, normalize=normalize)

    def forward(self, enhanced, low):
        M_low = self.sobel(low)                     # 参考：输入暗图梯度幅值场（常量，无梯度）
        M_enh = self.sobel(enhanced)                # 待约束：增强图梯度幅值场（有梯度）
        diff = (M_low - M_enh).abs()                # (4.12) 逐元素 L1
        # 对 (C,H,W) 求平均 = 1/(3N) Σ_c Σ_i；外层再对 batch 平均（公式为单图，训练 batch 取均值）
        return diff.mean(dim=(1, 2, 3)).mean()

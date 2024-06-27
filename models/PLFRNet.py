import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os
from models.pvtv2_encoder import pvt_v2_b2
from models.pvtv2_encoder import pvt_v2_b1
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class PLFRNet(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(PLFRNet, self).__init__()

        self.rgb_swin = pvt_v2_b2()
        self.depth_swin = pvt_v2_b1()

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.mefr_1 = MEF(512,1344)
        self.mefr_2 = MEF(320,960)
        self.mefr_3 = MEF(128,512)
        self.mefr_4 = MEF(64,256)

        self.mefd_1 = MEF(512,1344)
        self.mefd_2 = MEF(320,960)
        self.mefd_3 = MEF(128,512)
        self.mefd_4 = MEF(64,256)

        self.camf_1 = CAMF(1024)
        self.camf_2 = CAMF(640)
        self.camf_3 = CAMF(256)
        self.camf_4 = CAMF(128)

        self.rie_2 = RIE(128)
        self.rie_3 = RIE(64)
        self.rie_4 = RIE(64)

        self.up_edge4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_1 =  nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=320, kernel_size=1, bias=False),
            nn.BatchNorm2d(320),
            nn.GELU(),
            self.upsample2
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=640, out_channels=128,  kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64,  kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.pred_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1,  kernel_size=3,padding=1, bias=True),
            )
        self.pred_2 = nn.Conv2d(64, 1, kernel_size=3,stride=1)
        self.pred_3 = nn.Conv2d(128, 1, kernel_size=3,stride=1)
        self.pred_4 = nn.Conv2d(320, 1, kernel_size=3,stride=1)



    def forward(self,x,d):

        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r1 = rgb_list[3]
        r2 = rgb_list[2]
        r3 = rgb_list[1]
        r4 = rgb_list[0]

        d1 = depth_list[3]
        d2 = depth_list[2]
        d3 = depth_list[1]
        d4 = depth_list[0]

        r1 = self.mefr_1(r1,r2)
        r2 = self.mefr_2(r1,r2,r3)
        r3 = self.mefr_3(r2,r3,r4)
        r4 = self.mefr_4(r4,r3)

        d1 = self.mefd_1(d1,d2)
        d2 = self.mefd_2(d1,d2,d3)
        d3 = self.mefd_3(d2,d3,d4)
        d4 = self.mefd_4(d4,d3)
 
        camf_1 = self.camf_1(r1,d1)
        camf_2 = self.camf_2(r2,d2)
        camf_3 = self.camf_3(r3,d3)
        camf_4 = self.camf_4(r4,d4)

        fuse_1 = self.conv_1(camf_1)
        fuse_2 = self.rie_2(self.conv_2(torch.cat((fuse_1,camf_2), 1)))
        fuse_3 = self.rie_3(self.conv_3(torch.cat((fuse_2,camf_3), 1)))
        fuse_4 = self.rie_4(self.conv_4(torch.cat((fuse_3,camf_4), 1)))

        y1 = self.pred_1(fuse_4)
        y2 = F.interpolate(self.pred_2(fuse_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.pred_3(fuse_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.pred_4(fuse_1), size=384, mode='bilinear')



        return y1,y2,y3,y4



    def load_pre(self, pre_model_r,pre_model_d):

        pretrained_dict1 = torch.load(pre_model_r)
        pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in self.rgb_swin.state_dict()}
        self.rgb_swin.load_state_dict(pretrained_dict1)
        print(f"RGB PyramidVisionTransformerImpr loading pre_model ${pre_model_r}")

        pretrained_dict = torch.load(pre_model_d)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.depth_swin.state_dict()}
        self.depth_swin.load_state_dict(pretrained_dict)
        print(f"Depth PyramidVisionTransformerImpr loading pre_model ${pre_model_d}")



class MEF(nn.Module):
    def __init__(self, num_channels,int_channels, ratio=8):
        super(MEF, self).__init__()

        self.conv = nn.Conv2d(int_channels, num_channels,kernel_size=1,stride=1)
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.SA_Enhance = SpatialAttention()
        self.ca = ChannelAttention(int_channels)
    def forward(self, in1, in2=None, in3=None):
        if in2 !=None and in3 !=None:
            in1 = F.interpolate(in1, size=in2.size()[2:],mode='bilinear')
            in3 = F.interpolate(in3, size=in2.size()[2:], mode='bilinear')

        elif in2!=None and in3==None:
            in2 = F.interpolate(in2, size=in1.size()[2:],mode='bilinear')
            in3 = in1


        x = torch.cat((in1, in2, in3), 1)
        sa = self.SA_Enhance(x)
        sa = self.sa_conv(sa)
        out = self.ca(x.mul(sa))
        out = self.conv(out) * sa

        return out


class CAMF(nn.Module):
    def __init__(self, infeature):
        super(CAMF, self).__init__()
        self.SA_Enhance = SpatialAttention()
        self.channel_attention = ChannelAttention(infeature)
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.fuse = nn.Conv2d(infeature,infeature//2,kernel_size=1,stride=1)
    def forward(self, r, d):


        mul_fuse = r * d
        add_fuse = r + d
        all_fuse = torch.cat([mul_fuse, add_fuse], dim=1)
        sa = self.SA_Enhance(all_fuse)
        sa = self.conv(sa)
        ca = self.channel_attention(all_fuse.mul(sa))
        fuse = self.fuse(ca) * sa

        return fuse

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        mip = min(8,in_planes // ratio)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        ca = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = x * self.sigmoid(ca)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
            
class RIE(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.catt = Attention(dim,dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.catt(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x

class Attention(nn.Module):
    def __init__(self, inp, oup, ratio=16):
        super(Attention, self).__init__()

        self.SA_Enhance = SpatialAttention()
        mip = min(8, inp // ratio)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, inp, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb):
        x = rgb
        ca = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out_ca = x * self.sigmoid(ca)
        out_sa = self.SA_Enhance(out_ca)
        out = x.mul(out_sa)
        return out

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
        
        
        
if __name__=="__main__":

	import torch
	from thop import profile

	model = PLFRNet()

	a = torch.randn(1, 3, 384, 384)
	b = torch.randn(1, 3, 384, 384)
	flops, params = profile(model, (a,b))
	print('flops: ', flops, 'params: ', params)
	print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

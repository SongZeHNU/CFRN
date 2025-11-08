from sys import modules
from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit_encoder import vitencoder 
from torchvision import models
from config import get_config
import math
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import BatchNorm2d as bn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange
from lib.swin_transformer import Transformer, TransformerDecoder
from lib.ifa import ifa_simfpn

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv0_3 = nn.Conv2d(dim, dim, 3, padding=4, dilation=4)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv1_3 = nn.Conv2d(dim, dim, 3, padding=8, dilation=8)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv2_3 = nn.Conv2d(dim, dim, 3, padding=18, dilation=18)

        self.conv3 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_0 = self.conv0_3(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_1 = self.conv1_3(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_2 = self.conv2_3(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)
        #self.conv3(attn)

        return attn



class SpatialAttention(nn.Module):
    def __init__(self, d_model, channel):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

        self.GN = nn.GroupNorm(32, d_model)
        self.proj_3 = nn.Conv2d(d_model, 4*d_model, 1)
        self.activation2 = nn.GELU()

        self.proj_4 = nn.Conv2d(4*d_model, channel, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = self.GN(x)
        shorcut = x.clone()
        x = self.proj_3(x)
        x = self.activation2(x)
        x = self.proj_4(x)
        return x

class inrDecoder2(nn.Module):
    def __init__(self, inner_planes=32, num_classes=1, sync_bn=False, dilations=(12, 24, 36), 
                pos_dim=12, ultra_pe=False, unfold=False, no_aspp=False,
                local=False, stride=1, learn_pe=False, require_grad=True, num_layer=2):
        super(inrDecoder2, self).__init__()


        self.ifa = ifa_simfpn(ultra_pe=ultra_pe, pos_dim=pos_dim, sync_bn=sync_bn, num_classes=num_classes, local=local, unfold=unfold, stride=stride, learn_pe=learn_pe, require_grad=require_grad, num_layer=num_layer)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2, x3, x4, x5):

        context = []
        h, w = x5.shape[-2], x5.shape[-1]
        target_feat = [x1, x2, x3, x4, x5]
        for i, feat in enumerate(target_feat):
            context.append(self.ifa(feat, size=[h, w], level=i+1))

        context = torch.cat(context, dim=-1).permute(0,2,1)
        res = context.view(context.shape[0], -1, h, w)

        return res


class Network(nn.Module):
    def __init__(self, args, channel=32):
        super(Network, self).__init__()
        
        config = get_config(args)
        self.swin = vitencoder(config, img_size=args.trainsize)
        self.swin.load_from(config)


        self.eem1_1 = SpatialAttention(128,channel)     #channel
        self.eem2_1 = SpatialAttention(256,channel) 
        self.eem3_1 = SpatialAttention(512,channel)
        self.eem4_1 = SpatialAttention(1024,channel)
        self.eem5_1 = SpatialAttention(1024,channel)
        

        self.ND = inrDecoder2(channel)
        
       
        sumchannel = 5 * (channel + 34)   #34

        
        self.imnet2 = nn.Sequential(
            nn.Conv2d(sumchannel, sumchannel, 1),
            nn.Conv2d(sumchannel, 256, 1), nn.GroupNorm(32,256), nn.GELU(),
            nn.Conv2d(256, 256, 1), nn.GroupNorm(32,256),
            nn.Conv2d(256, 1, 1)
        )



   
    def forward(self, x):

        x = self.swin(x)


        x1 = x[4]                        # bs, 128, 96, 96
        _, pix, channel = x1.size()
        x1 = x1.transpose(1,2)
        x1 = x1.view(-1, channel, int(math.sqrt(pix)), int(math.sqrt(pix))) 

        x2 = x[0]                        # bs, 256, 48, 48    
        _, pix, channel = x2.size()
        x2 = x2.transpose(1,2)
        x2 = x2.view(-1, channel, int(math.sqrt(pix)), int(math.sqrt(pix))) 

        x3 = x[1]                        # bs, 512, 24, 24
        _, pix, channel = x3.size()
        x3 = x3.transpose(1,2)
        x3 = x3.view(-1, channel, int(math.sqrt(pix)), int(math.sqrt(pix))) 

        x4 = x[2]                       # bs, 1024, 12, 12
        _, pix, channel = x4.size()
        x4 = x4.transpose(1,2)
        x4 = x4.view(-1, channel, int(math.sqrt(pix)), int(math.sqrt(pix))) 

        x5 = x[3]                         # b, 1024, 12, 12s
        _, pix, channel = x5.size()
        x5 = x5.transpose(1,2)
        x5 = x5.view(-1, channel, int(math.sqrt(pix)), int(math.sqrt(pix))) 

        
      
        b1 = self.eem1_1(x1)        
        b2 = self.eem2_1(x2)     
        b3 = self.eem3_1(x3)      
        b4 = self.eem4_1(x4)   
        b5 = self.eem5_1(x5)

  
        S_g = self.ND(b5, b4, b3, b2, b1)

        S_g = self.imnet2(S_g)
        S_g1 = F.interpolate(S_g, scale_factor=4, mode='bilinear')  
        S_g_pred = S_g1
      




        return S_g_pred, S_g

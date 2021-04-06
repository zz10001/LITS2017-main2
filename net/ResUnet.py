"""

基础网络脚本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.checkpoint import checkpoint
import os



class encoder_stage(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(encoder_stage,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(ch_out),
            nn.PReLU(ch_out),

            nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=2, dilation=2),
            #不加空洞卷积
            # nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(ch_out),
            nn.PReLU(ch_out),

            nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=4, dilation=4),
            # nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(ch_out),
            nn.PReLU(ch_out),
            
            )
    
    def forward(self,x):
        x = self.conv(x)
        return x

class decoder_stage(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(decoder_stage,self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, 1, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.PReLU(ch_out),

            nn.Conv3d(ch_out, ch_out, 3, 1, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.PReLU(ch_out),

            nn.Conv3d(ch_out, ch_out, 3, 1, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.PReLU(ch_out)
    )

    def forward(self,x):
        x = self.decoder(x)
        return x


class down_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(down_conv,self).__init__()
        self.down = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 2, 2),
            nn.BatchNorm3d(ch_out),
            nn.PReLU(ch_out),
        )

    def forward(self,x):
        x = self.down(x)
        return x
# class down_conv_(nn.Module):
#     def __init__(self,ch_in,ch_out):
#         super(down_conv_,self).__init__()
#         self.down = nn.Sequential(
#             nn.Conv3d(ch_in, ch_out, 3, 1 , padding=1),
#             nn.PReLU(ch_out),
#         )
#     def forward(self,x):
#         x = self.down(x)
#         return x
    


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out, 2, 2),
            nn.BatchNorm3d(ch_out),
            nn.PReLU(ch_out),

        )

    def forward(self,x):
        x = self.up(x)
        return x

class _map(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(_map,self).__init__()
        self.map_=nn.Sequential(
            nn.Conv3d(ch_in,ch_out,1,1),
        )
    def forward(self,x):
        x = self.map_(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm3d(F_int),
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        
        self.relu = nn.PReLU()
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



class DialResUNet(nn.Module):
    def __init__(self,training):
        super().__init__()

        self.training = training
        self.Sigmoid = nn.Sigmoid()

        self.encoder_stage1 = encoder_stage(ch_in=1,ch_out=32)
        self.encoder_stage2 = encoder_stage(ch_in=64,ch_out=64)
        self.encoder_stage3 = encoder_stage(ch_in=128,ch_out=128)
        self.encoder_stage4 = encoder_stage(ch_in=256,ch_out=256)
        # self.encoder_stage5 = encoder_stage(ch_in=256,ch_out=256)

        # self.decoder_stage1 = decoder_stage(ch_in=128+128,ch_out=128)
        self.decoder_stage2 = decoder_stage(ch_in=256,ch_out=128)
        self.decoder_stage3 = decoder_stage(ch_in=128,ch_out=64)
        self.decoder_stage4 = decoder_stage(ch_in=64,ch_out=32)

        self.down_conv1 = down_conv(ch_in=32,ch_out=64)
        self.down_conv2 = down_conv(ch_in=64,ch_out=128)
        self.down_conv3 = down_conv(ch_in=128,ch_out=256)
        # self.down_conv4 = down_conv(ch_in=128,ch_out=256)

        # self.up_conv2 = up_conv(ch_in=256,ch_out=128)
        self.up_conv3 = up_conv(ch_in=256,ch_out=128)
        self.up_conv4 = up_conv(ch_in=128,ch_out=64)
        self.up_conv5 = up_conv(ch_in=64,ch_out=32)

        self.map5 = _map(ch_in=32,ch_out=2)
        self.map4 = _map(ch_in=64,ch_out=2)
        self.map3 = _map(ch_in=128,ch_out=2)
        self.map2 = _map(ch_in=256,ch_out=2)
        # self.map1 = _map(ch_in=256,ch_out=1)

        # self.att5 = Attention_block(128,128,64)
        self.att4 = Attention_block(128,128,64)
        self.att3 = Attention_block(64,64,32)
        self.att2 = Attention_block(32,32,16)

    def forward(self, inputs):
        # print('input',inputs.shape)

        long_range1 = self.encoder_stage1(inputs) + inputs  #16 256*256
        # print('long_range1',long_range1.shape)

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1  #32 128*128
        long_range2 = F.dropout(long_range2, 0.3, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2  #64*64*64
        long_range3 = F.dropout(long_range3, 0.3, self.training)

        short_range3 = self.down_conv3(long_range3) #128 32 32

        # print('short_range3',short_range3.shape)

        outputs = self.encoder_stage4(short_range3) + short_range3  #128 32 32 
        outputs = F.dropout(outputs, 0.3, self.training)

        output1 = self.map2(outputs)
        output1 = nn.functional.interpolate(output1,scale_factor=(8,8,8),mode='trilinear')
        # output1 = self.Sigmoid(output1)

        short_range6 = self.up_conv3(outputs)    #128-->64  64 64*64
        # print('short_eange6',short_range6.shape)

        short_range6 = self.att4(g=short_range6,x=long_range3)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6 #64*64 64
        outputs = F.dropout(outputs, 0.3, self.training)

        output2 = self.map3(outputs)
        output2 = nn.functional.interpolate(output2,scale_factor=(4,4,4),mode='trilinear')
        # output2 = self.Sigmoid(output2)

        short_range7 = self.up_conv4(outputs)  #32 128 128

        short_range7 = self.att3(g=short_range7,x=long_range2)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)

        output3 = self.map4(outputs)
        output3 = nn.functional.interpolate(output3,scale_factor=(2,2,2),mode='trilinear')
        # output3 = self.Sigmoid(output3)

        short_range8 = self.up_conv5(outputs) #16 256 256

        short_range8 = self.att2(g=short_range8,x=long_range1)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map5(outputs)
        output4 = nn.functional.interpolate(output4,scale_factor=(1,1,1),mode='trilinear')
        # output4 = self.Sigmoid(output4)
        # print('output4',output4.shape)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


net = DialResUNet(training=True)
net.apply(init)

# 输出数据维度检查
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
net = net.cuda()
data = torch.randn((2, 1, 64, 128, 128)).cuda()
res = net(data)
for item in res:
    print(item.size())

# 计算网络参数
num_parameter = .0
for item in net.modules():

    if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d):
        num_parameter += (item.weight.size(0) * item.weight.size(1) *
                          item.weight.size(2) * item.weight.size(3) * item.weight.size(4))

        if item.bias is not None:
            num_parameter += item.bias.size(0)

    elif isinstance(item, nn.PReLU):
        num_parameter += item.num_parameters


print(num_parameter)


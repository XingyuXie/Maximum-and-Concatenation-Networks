import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb



class MCN_block(nn.Module):
    def __init__(self, xs_dim, out_dim, x_strides =[2,2,2], bn_pool_flag = True):
        super(MCN_block, self).__init__()
        # xs_dim contains three parts:
        # x0_dim: the dimension of X0
        # xk_dim: the dimension of the previous layer
        # xkk_dim: the dimension of x_\hat{k}
        #--------------------------------------
        # out_dim contains three parts:
        # upperpart_out_dim: the dimension of the upper part of concatenation 
        # lowerpart_out_dim: the dimension of the lower part of concatenation 
        #--------------------------------------
        # x_strides contains three parts:
        # x0_stride: the conv stride for X0 
        # xk_stride: the conv stride for Xk
        #--------------------------------------
        # bn_pool_flag: Whether to perform the BN+ReL before output
        # init size  tuple
        self.x0_dim = xs_dim[0]
        self.xkk_dim = xs_dim[1]
        self.xk_dim = xs_dim[2]
        self.upperpart_out_dim = out_dim[0]
        self.lowerpart_out_dim = out_dim[1]
        self.output_dim = out_dim[0]+out_dim[1]        
        self.x0_stride = x_strides[0]
        self.xkk_stride = x_strides[1]
        self.xk_stride = x_strides[2]
        self.bn_pool_flag = bn_pool_flag

        #self.gamma_alpha = gamma_alpha
        #

        
        self.conv_Ak = nn.Conv2d(self.xkk_dim, self.lowerpart_out_dim, kernel_size=3, stride = self.xkk_stride, padding=1)
        self.conv_tildeAk = nn.Conv2d(self.x0_dim, self.lowerpart_out_dim, kernel_size=3, stride = self.x0_stride, padding=1)
        self.conv_Wk = nn.Conv2d(self.xk_dim, self.lowerpart_out_dim, kernel_size=3, stride = self.xk_stride, padding=1)
        self.conv_Lk = nn.Conv2d(self.xk_dim, self.upperpart_out_dim, kernel_size=3, stride = self.xk_stride, padding=1)
        
        
        self.adaptiveScalar_in = nn.Parameter(torch.tensor(1e-2))
        self.adaptiveScalar_out = nn.Parameter(torch.tensor(1e-2))
        self.bn_f = nn.BatchNorm2d(self.upperpart_out_dim + self.lowerpart_out_dim)
        self.relu_f = nn.LeakyReLU(negative_slope=0.15, inplace=True)

    def forward(self, x_0, x_kk, x_k):
        # See the autograd section for explanation of what happens here.
        LkX = self.conv_Lk(x_k)
        WkX = self.conv_Wk(x_k)
        AkX = self.conv_Ak(x_kk)
        sigmaAkX = self.relu_f(AkX)
        tildeAkX = self.conv_tildeAk(x_0)
        #pdb.set_trace()
        low_part = torch.exp(tildeAkX*self.adaptiveScalar_in)*self.adaptiveScalar_out + torch.max(sigmaAkX,WkX)
        output = torch.cat((LkX, low_part), 1)
        Bn_output = self.bn_f(output)
        Act_Bn_output = self.relu_f(Bn_output)
        return Act_Bn_output if self.bn_pool_flag else output





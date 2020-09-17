import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb



class MCN_block(nn.Module):
    def __init__(self, x0_dim, input_dim, upperpart_out_dim, lowerpart_out_dim, x0_stride = 2, xk_stride =1, bn_pool_flag = True):
        super(MCN_block, self).__init__()
        # x0_dim: the dimension of X0
        # input_dim: the dimension of the previous layer
        # upperpart_out_dim: the dimension of the upper part of concatenation 
        # lowerpart_out_dim: the dimension of the lower part of concatenation 
        # x0_stride: the conv stride for X0 
        # xk_stride: the conv stride for Xk
        # bn_pool_flag: Whether to perform the BN+ReL before output
        # init size  tuple
        self.input_dim = input_dim
        self.upperpart_out_dim = upperpart_out_dim
        self.lowerpart_out_dim = lowerpart_out_dim
        self.output_dim = upperpart_out_dim+lowerpart_out_dim
        self.x0_dim = x0_dim
        self.x0_stride = x0_stride
        self.xk_stride = xk_stride
        self.bn_pool_flag = bn_pool_flag
        #self.gamma_alpha = gamma_alpha
        #

        
        self.conv_Ak = nn.Conv2d(self.x0_dim, self.lowerpart_out_dim, kernel_size=3, stride = self.x0_stride, padding=1)
        self.conv_tildeAk = nn.Conv2d(self.x0_dim, self.lowerpart_out_dim, kernel_size=3, stride = self.x0_stride, padding=1)
        self.conv_Wk = nn.Conv2d(self.input_dim, self.lowerpart_out_dim, kernel_size=3, stride = self.xk_stride = xk_stride, padding=1)
        self.conv_Lk = nn.Conv2d(self.input_dim, self.upperpart_out_dim, kernel_size=3, stride = self.xk_stride = xk_stride, padding=1)
        
        
        self.adaptiveScalar = nn.Parameter(torch.tensor(1e-1))
        self.bn_f = nn.BatchNorm2d(self.upperpart_out_dim + self.lowerpart_out_dim)
        self.relu_f = nn.LeakyReLU(negative_slope=0.15, inplace=True)

    def forward(self, input_x_0, input_x_k):
        # See the autograd section for explanation of what happens here.
        LkX = self.conv_Lk(input_x_k)
        WkX = self.conv_Wk(input_x_k)
        AkX = self.conv_Ak(input_x_0)
        sigmaAkX = self.relu_f(AkX)
        tildeAkX = self.conv_tildeAk(input_x_0)
        #pdb.set_trace()
        low_part = torch.exp(tildeAkX*self.adaptiveScalar) + torch.max(sigmaAkX,WkX)
        #low_part = torch.exp(tildeAkX*0.5) + torch.max(sigmaAkX,WkX)
        output = torch.cat((LkX, low_part), 1)
        Bn_output = self.bn_f(output)
        Act_Bn_output = self.relu_f(Bn_output)
        return Act_Bn_output if self.bn_pool_flag else output





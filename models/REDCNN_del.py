import torch
from torch.autograd import Variable
import torch.nn as nn


class REDCNN(nn.Module):
    def __init__(self, cube_len=64):
        
        super(REDCNN, self).__init__()
        
        self.cube_len = cube_len
        self.code_len = cube_len * 8
        self.feature_maps = 96
        self.kernel_size = 5
        self.stride = 1

        #Contracting path:
        
        self.enc_1 = nn.Sequential(
            nn.Conv3d(1, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
            nn.ReLU()
        )
        
        self.enc_2 = nn.Sequential(
            nn.Conv3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
            nn.ReLU()
        )
        
        self.enc_3 = nn.Sequential(
            nn.Conv3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
            nn.ReLU()
        )
        
        self.enc_4 = nn.Sequential(
            nn.Conv3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
            nn.ReLU()
        )
        
        self.enc_5 = nn.Sequential(
            nn.Conv3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
            nn.ReLU()
        )


        #Expansive path
        
        self.dec_1_deconv = nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride)
        self.dec_1_ReLU = nn.ReLU()
        
        self.dec_2_deconv = nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride)
        self.dec_2_ReLU = nn.ReLU()

        self.dec_3_deconv = nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride)
        self.dec_3_ReLU = nn.ReLU()

        self.dec_4_deconv = nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride)
        self.dec_4_ReLU = nn.ReLU()

        self.dec_5_deconv = nn.ConvTranspose3d(self.feature_maps, 1, kernel_size = self.kernel_size, stride = self.stride)
        self.dec_5_ReLU = nn.ReLU()

        
    
    def forward(self, x):
        #encoder path
        input_volume = x.clone()

        out1 = self.enc_1(x)
        
        out2 = self.enc_2(out1)
        feature_maps_2 = out2.clone()
        del out1

        out3 = self.enc_3(out2)        
        del out2

        out4 = self.enc_4(out3)
        feature_maps_4 = out4.clone()
        del out3

        out5 = self.enc_5(out4)
        del out4
         
        
        #decoder path
        out1 = self.dec_1_deconv(out5)
        out1 = out1 + feature_maps_4
        out1 = self.dec_1_ReLU(out1)
        del out5
        del feature_maps_4
        
        out2 = self.dec_2_deconv(out1)
        out2 = self.dec_2_ReLU(out2)
        del out1

        out3 = self.dec_3_deconv(out2)
        out3 = out3 + feature_maps_2
        out3 = self.dec_3_ReLU(out3)
        del out2
        del feature_maps_2

        out4 = self.dec_4_deconv(out3)
        out4 = self.dec_4_ReLU(out4)        
        del out3

        out5 = self.dec_5_deconv(out4)
        out5 = out5 + input_volume
        out5 = self.dec_5_ReLU(out5)
        del out4
        del input_volume  
        
        return out5 
        

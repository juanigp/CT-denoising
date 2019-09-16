import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class REDCNN_(nn.Module):
    def __init__(self, cube_len=64):
        
        super(REDCNN_, self).__init__()
        
        self.cube_len = cube_len
        self.code_len = cube_len * 8
        self.feature_maps = 96
        self.kernel_size = 5
        self.stride = 1

        #Contracting path:
        
        self.enc_1 = nn.Sequential(
            nn.Conv3d(1, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
        )
        
        self.enc_2 = nn.Sequential(
            nn.Conv3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
        )
        
        self.enc_3 = nn.Sequential(
            nn.Conv3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
        )
        
        self.enc_4 = nn.Sequential(
            nn.Conv3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
        )
        
        self.enc_5 = nn.Sequential(
            nn.Conv3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
        )


        #Expansive path
        
        self.dec_1_deconv = nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride)
        
        self.dec_2_deconv = nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride)

        self.dec_3_deconv = nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride)

        self.dec_4_deconv = nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride)

        self.dec_5_deconv = nn.ConvTranspose3d(self.feature_maps, 1, kernel_size = self.kernel_size, stride = self.stride)

        
    
    def forward(self, x):
        #encoder path
        input_volume = x.clone()

        out = F.relu( self.enc_1(x) )
        
        out = F.relu( self.enc_2(out) )
        feature_maps_2 = out.clone()

        out = F.relu(  self.enc_3(out) )       
        
        out = F.relu( self.enc_4(out) )
        feature_maps_4 = out.clone()
        
        out = F.relu( self.enc_5(out) )
         
        
        #decoder path
        out = F.relu( self.dec_1_deconv(out) + feature_maps_4 )
        
        out = F.relu( self.dec_2_deconv(out) )

        out = F.relu( self.dec_3_deconv(out) + feature_maps_2 )

        out = F.relu( self.dec_4_deconv(out) )  

        out = F.relu( self.dec_5_deconv(out) + input_volume )
        
        return out 
        

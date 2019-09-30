import torch
from torch.autograd import Variable
import torch.nn as nn


class REDCNN(nn.Module):
    def __init__(self):
        
        super(REDCNN, self).__init__()
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

        out = self.enc_1(x)
        
        out = self.enc_2(out)
        feature_maps_2 = out.clone()

        out = self.enc_3(out)        
        
        out = self.enc_4(out)
        feature_maps_4 = out.clone()
        
        out = self.enc_5(out)
         
        
        #decoder path
        out = self.dec_1_deconv(out)
        out = out + feature_maps_4
        out = self.dec_1_ReLU(out)
        
        out = self.dec_2_deconv(out)
        out = self.dec_2_ReLU(out)

        out = self.dec_3_deconv(out)
        out = out + feature_maps_2
        out = self.dec_3_ReLU(out)

        out = self.dec_4_deconv(out)
        out = self.dec_4_ReLU(out)        

        out = self.dec_5_deconv(out)
        out = out + input_volume
        out = self.dec_5_ReLU(out)  
        
        return out 
        

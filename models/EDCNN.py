import torch
from torch.autograd import Variable
import torch.nn as nn


class EDCNN(nn.Module):
    def __init__(self):
        
        super(EDCNN, self).__init__()
        self.feature_maps = 32
        self.kernel_size = (3, 5, 5)
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
        

        #Expansive path
        self.dec_1 = nn.Sequential(
            nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
            nn.ReLU()
        )
        
        self.dec_2 = nn.Sequential(
            nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
            nn.ReLU()
        )

        self.dec_3 = nn.Sequential(
            nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
            nn.ReLU()
        )

        self.dec_1 = nn.Sequential(
            nn.ConvTranspose3d(self.feature_maps, self.feature_maps, kernel_size = self.kernel_size, stride = self.stride),
            nn.ReLU()
        )

        self.dec_4 = nn.ConvTranspose3d(self.feature_maps, 1, kernel_size = self.kernel_size, stride = self.stride)
        
    
    def forward(self, x):
        #encoder path
        out = self.enc_1(x)
        out = self.enc_2(out)
        out = self.enc_3(out)     
        out = self.enc_4(out)
        
        #decoder path
        out = self.dec_1(out)
        out = self.dec_2(out)
        out = self.dec_3(out)
        out = self.dec_4(out)

        return out 
        

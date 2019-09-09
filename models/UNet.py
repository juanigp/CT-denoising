import torch
from torch.autograd import Variable
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, cube_len=64):
        
        super(UNet, self).__init__()
        
        self.cube_len = cube_len
        self.code_len = cube_len * 8
        
        #Contracting path:
        
        self.enc_1 = nn.Sequential(
            nn.Conv3d(1, self.cube_len, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.cube_len),
            nn.ReLU()
        )
        
        
        self.enc_2 = nn.Sequential(
            nn.Conv3d(self.cube_len, self.cube_len * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.cube_len * 2),
            nn.ReLU()        
        )
        
        self.enc_3 = nn.Sequential(
            nn.Conv3d(self.cube_len * 2, self.cube_len * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.cube_len * 4),
            nn.ReLU()        
        ) 
        
        self.enc_4 = nn.Sequential(
            nn.Conv3d(self.cube_len * 4, self.code_len, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.code_len),
            nn.ReLU()        
        ) 
        
        self.enc_5 = nn.Sequential(
            nn.Conv3d(self.code_len, self.code_len, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.code_len),
            nn.ReLU()        
        )  
        
        self.enc_6 = nn.Sequential(
            nn.Conv3d(self.code_len, self.code_len, kernel_size = 4, stride = 2, padding = 1),
            #cant batch norm when features are 1x1x1
            #nn.BatchNorm3d(self.code_len),
            nn.ReLU()        
        )
        
        #Expansive path
        
        self.dec_1 = torch.nn.Sequential(
            nn.ConvTranspose3d(self.code_len, self.code_len, kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm3d(self.code_len),
            nn.ReLU()
        )
        
        #According to the paper this layer also has dropout
        self.dec_2 = torch.nn.Sequential(
            nn.ConvTranspose3d( (self.code_len) * 2, self.code_len, kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm3d(self.code_len),
            nn.ReLU()
        )        
        
        self.dec_3 = torch.nn.Sequential(
            nn.ConvTranspose3d( (self.code_len) * 2, (self.cube_len * 4) , kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm3d((self.cube_len * 4)),
            nn.ReLU()
        )        
        
        self.dec_4 = torch.nn.Sequential(
            nn.ConvTranspose3d( (self.cube_len * 4) * 2, (self.cube_len * 2) , kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm3d((self.cube_len * 2)),
            nn.ReLU()
        )

        self.dec_5 = torch.nn.Sequential(
            nn.ConvTranspose3d( (self.cube_len * 2) * 2, self.cube_len, kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm3d(self.cube_len),
            nn.ReLU()
        )
        
        self.dec_6 = torch.nn.Sequential(
            nn.ConvTranspose3d( (self.cube_len) * 2, 1, kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm3d(1),
            nn.ReLU()
        )
        
    
    
    def forward(self, x):
        #downconvolutions
        out = self.enc_1(x)
        feature_map_1  = out.clone()
        
        out = self.enc_2(out)
        feature_map_2 = out.clone()

        out = self.enc_3(out)
        feature_map_3 = out.clone()        
        
        out = self.enc_4(out)
        feature_map_4 = out.clone()
        
        out = self.enc_5(out)
        feature_map_5 = out.clone()
         
        #code
        out = self.enc_6(out)
        
        #upconvolutions
        out = self.dec_1(out)
        dec_2_in = torch.cat((out, feature_map_5), 1)
        
        out = self.dec_2(dec_2_in)
        dec_3_in = torch.cat((out, feature_map_4), 1)
        
        out = self.dec_3(dec_3_in)
        dec_4_in = torch.cat((out, feature_map_3), 1)
        
        out = self.dec_4(dec_4_in)
        dec_5_in = torch.cat((out, feature_map_2), 1)
        
        out = self.dec_5(dec_5_in)
        dec_6_in = torch.cat((out, feature_map_1), 1)
        
        out = self.dec_6(dec_6_in)
        
        return out 
        

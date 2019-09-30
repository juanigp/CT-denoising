#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:51:00 2018

@author: felix

to train and everything:
"""
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data.dataset import random_split
from torchsummary import summary
import time
import numpy as np

from tqdm import tqdm 
from modules import patches 
import os
from utils import nn_utils

BSIZE = 16 # batch-size, according to GPU etc.
PERCENT_DATA = 5 # option, not to use all samples (value between 1 and 100) 
EPOCHS = 10 
LR = 0.0001

CUBELEN = 2 # parameter of net
FOLDER_PATCH = '19x39x39'
DATA_SUBSTR = '360_FBPPhil'
TARGET_SUBSTR = '500FBP'
PERCENT_TRAIN = 95 
TEST_BLOCKS = 1
CUDA = True
SLEEPTIME = 0.0001

class TestConv1(nn.Module):
    def __init__(self,cubelen):
        super(TestConv1, self).__init__()
        ks1 = (3,5,5)
        self.enc = nn.Sequential(
            nn.Conv3d(1,        cubelen,kernel_size=ks1),
            nn.LeakyReLU(),
            nn.Conv3d(cubelen,  cubelen,kernel_size=ks1),
            nn.LeakyReLU(),
            nn.Conv3d(cubelen,  cubelen*2,kernel_size=ks1),
            nn.LeakyReLU(),
            nn.Conv3d(cubelen*2,  cubelen*2,kernel_size=ks1),
            nn.LeakyReLU(),
            nn.Conv3d(cubelen*2,  cubelen*4,kernel_size=ks1),
            nn.LeakyReLU(), 
            nn.Conv3d(cubelen*4,  cubelen*4,kernel_size=ks1),
            nn.LeakyReLU(),
            nn.Conv3d(cubelen*4,  cubelen*4,kernel_size=ks1),
            #nn.LeakyReLU(),
            nn.Conv3d(cubelen*4,1,kernel_size=1)
            )
        
        # number of voxels that the net reduces (could be neagative as well)
        nom_pad = (7,14,14)
        # number of voxels one should reduce to evaluate result (>=nom_pad)
        eff_pad = (7,14,14)
        
        self.padO =  (eff_pad[0]-nom_pad[0],eff_pad[1]-nom_pad[1],eff_pad[2]-nom_pad[2])
        self.padT = eff_pad
    
    def forward(self, x):
        out = self.enc(x)
        return out
     
class cropMSELoss(nn.MSELoss):
    def __init__(self,  size_average=True, reduce=True,lim1=-100,lim2=500):
        super(cropMSELoss,self).__init__(size_average, reduce)
        self.lim1 = lim1
        self.lim2 = lim2
        
        
    def forward(self, input,target):
        assert not target.requires_grad
        
        input = clampalpha(input,self.lim1,self.lim2)
        target = clampalpha(target,self.lim1,self.lim2)
        return F.mse_loss(input,target,size_average=self.size_average, reduce=self.reduce)
        
def clampalpha(data,l1,l2,alpha=0.5):
    m = nn.LeakyReLU(alpha)
    data = m(data-l1)+l1
    data = l2-m(l2-data)
    return data
    
            
    
class CNN(nn_utils.NN):
    _data_substr = DATA_SUBSTR
    _target_substr = TARGET_SUBSTR
    _bsize = BSIZE
        
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and
                                   CUDA else "cpu")
        # network init:
        self.module = TestConv1(CUBELEN).to(self.device)          
        self.criterion = cropMSELoss()        
        self.optimizer = optim.Adam(self.module.parameters(), lr = LR)
        # data loader:
        PATH_PATCH = os.path.join('data/patches',FOLDER_PATCH)
        if not os.path.exists(PATH_PATCH):
           print('Could not find patches, try running make_patches')
           patches.make_patches()
        data = nn_utils.Dataset(PATH_PATCH,self._data_substr,self._target_substr,PERCENT_DATA)
        patch_shape = data[0][0].shape
        
        
        print('---------- Networks architecture Generator -------------')
        summary(self.module,  (1, patch_shape[1],patch_shape[2],patch_shape[3]))
        
        # make loaders:
        numberTrain = int(len(data)*PERCENT_TRAIN/100)
        train_data,test_data = random_split(data,(numberTrain,len(data)-numberTrain))
        self.train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BSIZE, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BSIZE, shuffle=True)
        self.dataset_vol = None
    
    def test_loss(self,length,loader):
        ''' compute loss on independent sample'''
        lossOutput,lossData = 0,0
        with torch.no_grad():
            for idx,(data,target) in enumerate(loader):
                target = nn_utils.crop(target,self.module.padT).to(self.device)
                data = data.to(self.device)
                output = self.module(data)
                output = nn_utils.crop(output,self.module.padO)
                lossOutput += self.criterion(output,target).item()/np.prod(output.shape)
                data = nn_utils.crop(data,self.module.padT)
                lossData += self.criterion(data,target).item()/np.prod(output.shape) 
                if idx==length-1:
                    break
        return (lossOutput,lossData)
    
    def print_results(self):
        ''' print loss from two independent samples (train and test)'''
        length = TEST_BLOCKS 
        length = min(min(length,len(self.test_loader)),len(self.train_loader))
        self.module.eval()
        lossTest,lossTest0 = self.test_loss(length,self.test_loader)
        lossTrain,lossTrain0 = self.test_loss(length,self.train_loader)
        imprTrain = lossTrain0/lossTrain 
        imprTest = lossTest0/lossTest
        #tqdm.write("\t Impr. (train/test): %.5f/%.5f = %.2f%% "% (imprTrain,imprTest,imprTrain/imprTest*100))
        self.train_hist['imprTrain'].append(imprTrain)
        self.train_hist['imprTest'].append(imprTest)
        #time.sleep(SLEEPTIME)
        self.module.train() 
        
    def stats_and_screenshot(self):
        if self.dataset_vol == None:
            self._read_test_data()
            
            patch_shape = self.train_loader.dataset[0][0].shape[1:]
            self.dataset_vol = nn_utils.Dataset_volume(self._data_vol,
                        shape=patch_shape,padding=self.module.padT)
                 
            # compute eventually statistics of input and target over mask
            pad = self.module.padT
            self._data_tar = nn_utils.crop(self._data_tar,pad)
            self._data_vol = nn_utils.crop(self._data_vol,pad)
            self._data_mask = nn_utils.crop(self._data_mask,pad)
            nans = np.logical_or(np.isnan(self._data_tar),np.isnan(self._data_vol))
            nans = np.logical_or(nans,np.isnan(self._data_mask))
            
            self._data_tar[nans] = 0
            self._data_vol[nans] = 0
            self._data_mask[nans] = 0
            rmse = nn_utils.rmse(self._data_vol,self._data_tar,self._data_mask)
            psnr = nn_utils.psnr(self._data_vol,self._data_tar,self._data_mask)
            #ssim = nn_utils.ssim(self._data_vol,self._data_tar,self._data_mask)
            self.train_hist['rmse0'] = rmse
            self.train_hist['psnr0'] = psnr
            #self.train_hist['ssim0'] = ssim
        self._minT = 0
        self._maxT = 500
        
        self.apply_on_dataset(self.dataset_vol)
        output = self.dataset_vol.output
        path = os.path.join(self._write_path,'screenshots')
        epoch = len(self.train_hist['rmse'])
        nn_utils.saveAsPNG(path,'2output'+str(epoch+1).zfill(3)+'.png',output,self._minT,self._maxT)
        
        # compute eventually statistics of output and target over mask
        output = nn_utils.crop(output,self.module.padT)
        more_nans = np.isnan(output)
        self._data_mask[more_nans] = 0
        output[more_nans] = 0
        rmse = nn_utils.rmse(output,self._data_tar,self._data_mask)
        psnr = nn_utils.psnr(output,self._data_tar,self._data_mask)
        
        #ssim = nn_utils.ssim(output,self._data_tar,self._data_mask)
        self.train_hist['rmse'].append(rmse)
        self.train_hist['psnr'].append(psnr)
        #self.train_hist['ssim'].append(ssim)
        if len(self.train_hist['rmse'])>1:
            self.plot_from_hist('rmse')
            self.plot_from_hist('psnr')
            #self.plot_from_hist('ssim')
        
        
      

    def initialize_train_hist(self):
        ''' initialize all variables we want to log! '''
        self.train_hist['imprTrain'] =[]
        self.train_hist['imprTest'] =[]
        self.train_hist['loss'] =[]
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = [] 
        self.train_hist['rmse'] = []
        self.train_hist['psnr'] = []
        self.train_hist['ssim'] = []    
    
    def train(self):    
        if len(self.train_hist)==0:    
            self.initialize_train_hist()
        
        print('Training...')
        start_time = time.time()              
        for epoch in range(EPOCHS):
            self.module.train()
            epoch_start_time = time.time()
            pbarText = str(epoch+1).zfill(len(str(EPOCHS)))+ '/' +str(EPOCHS)
            pbar = tqdm(desc = pbarText,total=len(self.train_loader))
            sumLoss = 0
            for idx, (data,target) in enumerate(self.train_loader):                
                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                target = nn_utils.crop(target,self.module.padT)
                
                # update network:
                output = self.module(data)
                output = nn_utils.crop(output,self.module.padO)
                loss = self.criterion(output,target)
                self.train_hist['loss'].append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                sumLoss += loss.item()
                pbar.set_description(pbarText+ ' E=' + str(np.round(sumLoss/(idx+1))),refresh=False)
                pbar.update(1)
            pbar.close()
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            #self.print_results() 
            self.stats_and_screenshot()
            self.nnwrite()
        self.train_hist['total_time'].append(time.time() - start_time)
        print("\ntime/epoch: %.2f, total time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
               self.train_hist['total_time'][0]))
        self.module.eval()
        self.nnwrite()
        print("Training finished!") 
        
    def apply(self,volume,patch_shape=(0,0,0)): 
        ''' apply module on given numpy volume, using given patch_size'''
        if patch_shape[0]==0:   
            patch_shape = self.train_loader.dataset[0][0].shape[1:]
        patch_shape = np.minimum(volume.shape,patch_shape)
        dataset_vol = nn_utils.Dataset_volume(volume,shape=patch_shape,
                                              padding=self.module.padT)
        self.apply_on_dataset(dataset_vol)
        return dataset_vol.output
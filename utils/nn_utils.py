#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:16:35 2018

@author: felix
"""
import torch
import torch.utils.data
import random
import numpy as np
import imageio
import os.path
from shutil import copyfile
import pickle

from utils import volumeio
import scipy.ndimage

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class NN(object):
    """ NN: general class of a neuronal network to define interfaces for 
    utils-functions"""
    module = torch.nn.Module()
    device = torch.device("cpu")
    criterion = None 
    optimizer = None

    _write_path = None
    train_hist = {}
    _data_substr = '360_FBPPhil'
    _target_substr = '500FBP'
    
    _data_vol = None
    _data_tar = None
    _data_mask = None
    _bsize = 8
    
    def _make_path(self):
        # get all folders in output
        path = 'output'
        
        if not os.path.exists(path):
            os.makedirs(path)  
        subfolders = [int(f.name.split('_')[1]) 
            for f in os.scandir(path) 
            if f.is_dir() and f.name.startswith('NN')]
        subfolders.append(-1)
        newPath = 'NN_'+str(max(subfolders)+1).zfill(4) + \
            '_'+self.module.__class__.__name__+ \
            '_'+self.criterion.__class__.__name__
        self._write_path = os.path.join(path,newPath)
        os.makedirs(self._write_path)
        src = os.path.join('modules','CNN.py')
        dst = os.path.join(self._write_path,'CNN.py')
        copyfile(src, dst)
      
    def _get_path(self):
        path = 'output'   
        subfolders = [int(f.name.split('_')[1]) 
            for f in os.scandir(path) 
            if f.is_dir() and f.name.startswith('NN')]
        if len(subfolders)==0:
            raise ValueError('Cannot find state dict file.') 
        maxIndex = max(subfolders)
        folder = [f.name for f in os.scandir(path) 
            if f.is_dir() and f.name.startswith('NN_') 
            and int(f.name.split('_')[1])==maxIndex][0]
        self._write_path = os.path.join(path,folder)
        
    def nnwrite(self):
        ''' save database and history:'''
        if self._write_path == None:
            self._make_path()    
        torch.save(self.module.state_dict(), os.path.join(self._write_path,'Module_state_dict.pkl'))
        torch.save(self.optimizer.state_dict(), os.path.join(self._write_path,'Optimizer_state_dict.pkl'))
        with open(os.path.join(self._write_path,'History.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
        
    def nnread(self):
        ''' load database and history:'''
        if self._write_path==None:
            self._get_path()
        print('read network: ' + self._write_path)
        self.module.load_state_dict(torch.load(os.path.join(self._write_path,'Module_state_dict.pkl')))
        self.optimizer.load_state_dict(torch.load(os.path.join(self._write_path,'Optimizer_state_dict.pkl')))
        with open(os.path.join(self._write_path,'History.pkl'), 'rb') as f:
            self.train_hist = pickle.load(f)
    
    def _read_test_data(self):
        ''' prepare to apply module to volume and save center slice screenshots of inputs'''
        if self._write_path == None:
            self._make_path()
        path = os.path.join(self._write_path,'screenshots')
        if not os.path.exists(path):
            os.makedirs(path)
        pathIn = os.path.join('data','test') 
        
        # open data:
        substring = self._data_substr
        fn_list = [f for f in os.listdir(pathIn) if f.find(substring)>0]
        if len(fn_list)==0:
            raise ValueError('Cannot find test file in folder ' + pathIn +' containing substring: ' + substring)     
        fn_test = os.path.join(pathIn,fn_list[0])
        data_vol = volumeio.OpenXML(fn_test,kind='Slices',asNumpy=True)
        data_vol[np.isnan(data_vol)]=0
        #data_vol = torch.load(fn_test)
        centerZ = data_vol.shape[0]//2
        z = self.train_loader.dataset[0][0].shape[1]
        z0 = centerZ-z//2
        z1 = z0 + z
        self._data_vol = data_vol[z0:z1,:,:]
        saveAsPNG(path,'1_input.png',self._data_vol,percentile = 2)
        
        # open target:
        substring = self._target_substr
        fn_list = [f for f in os.listdir(pathIn) if f.find(substring)>0]
        if len(fn_list)==0:
            raise ValueError('Cannot find target file in folder ' + pathIn +' containing substring: ' + substring)     
        fn_test = os.path.join(pathIn,fn_list[0])
        data_tar = volumeio.OpenXML(fn_test,kind='Slices',asNumpy=True)
        data_tar[np.isnan(data_tar)]=0
        #data_tar = torch.load(fn_test)
        #self._data_tar = data_tar[z0:z1,:,:]
        self._data_tar = data_tar[z0:z1,:,:]
        self._minT = np.percentile(self._data_tar,2)
        self._maxT = np.percentile(self._data_tar,98)
        saveAsPNG(path,'0_target.png',self._data_tar,self._minT,self._maxT)
        
        #open mask:
        substring = 'Mask.'
        fn_list = [f for f in os.listdir(pathIn) if f.find(substring)>0]
        if len(fn_list)==0:
            raise ValueError('Cannot find mask file in folder ' + pathIn +' containing substring: ' +substring)     
        fn_test = os.path.join(pathIn,fn_list[0])
        data_mask = volumeio.OpenXML(fn_test,kind='Mask',asNumpy=True)
        
        #data_mask = torch.load(fn_test)
        #self._data_mask = (data_mask[z0:z1,:,:]>0)*1
        self._data_mask = (data_mask[z0:z1,:,:]>0)*1
        
    def plot_from_hist(self,value):
        data = self.train_hist[value]
        epoch = np.arange(len(data))+1
        
        # don't show it on screen...
        plt.ioff()
        fig = plt.figure(figsize=(16, 8))
        plt.xlim(0.9,max(epoch)+0.1)
        if max(data)>0:
            plt.ylim(0,max(data)*1.05)
        plt.plot(epoch,data,'.-')

        plt.title(value + ' on ' +self._write_path)
        plt.xlabel('epoch')
        plt.ylabel(value)
        path = os.path.join(self._write_path,value+'.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        plt.ion()
        
    def apply_on_dataset(self,dataset_vol):
        loader_vol=torch.utils.data.DataLoader(dataset=dataset_vol, batch_size=self._bsize)
        self.module.eval()
        dataset_vol.reset_output()
        with torch.no_grad():
            for idx,(data,idx_i) in enumerate(loader_vol):
                output = self.module(data.to(self.device))
                output = crop(output,self.module.padO)
                dataset_vol.reconstruct(output,idx_i)   
        
    def train(self):
        raise NotImplementedError

# dataset to process given numpy volume
class Dataset_volume(torch.utils.data.Dataset):
    def __init__(self, volume,shape=(19,39,39),padding=(0,0,0)):
        # tesselate volume:
        shape = np.array(shape)
        padding = np.array(padding)
        shape_vol = np.shape(volume)
        start = []
        for i in range(3):
            cstart = np.arange(0,shape_vol[i]-padding[i]*2,shape[i]-padding[i]*2)
            cstart = np.minimum(0,(shape_vol[i]-cstart-shape[i]))+cstart
            start.append(cstart)
        
        # get all combinations of positions
        gridx,gridy,gridz = np.meshgrid(start[1],start[0],start[2])
        self.patch_start = np.concatenate((gridy.reshape(-1,1),
                                     gridx.reshape(-1,1),
                                     gridz.reshape(-1,1)),axis=1)
        #self.require_crop = require_crop
        self.shape = shape
        self.offset1 = padding
        self.offset2 = shape-padding
        self.volume = volume
        self.output = np.zeros((0,0,0))
        self.eff_pad = padding

    def __len__(self): 
        return len(self.patch_start)
 
    def __getitem__(self, idx):
        pos = self.patch_start[idx]
        s = self.shape
        data = self.volume[pos[0]:pos[0]+s[0],pos[1]:pos[1]+s[1],pos[2]:pos[2]+s[2]]
        data = torch.tensor(data,dtype=torch.float32).unsqueeze(0)
        return (data,idx)
    
    def reset_output(self):
        self.output = np.zeros((0,0,0)) 
    
    def reconstruct(self,patch,idx):
        if len(self.output) == 0:
            self.output = np.zeros(self.volume.shape)
        o1 = self.offset1
        o2 = self.offset2
        for i in range(len(idx)):    
            pos = self.patch_start[idx[i]]
            self.output[pos[0]+o1[0]:pos[0]+o2[0],
                    pos[1]+o1[1]:pos[1]+o2[1],
                    pos[2]+o1[2]:pos[2]+o2[2]] = patch[i,0,:,:,:]         


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path,data_substring,target_substring,percent=100,shuffle=True):
        self.keys = []
        fn_test = [f for f in os.listdir(path) if f.find(data_substring)>0][0]
        idxLine = fn_test.find('_')        
        data_fn = {os.path.join(path,f) : f[0:idxLine] for f 
                  in os.listdir(path) 
                  if f.find(data_substring)>=0}
        self.keys = list(data_fn.keys())
        if shuffle:
            random.shuffle(self.keys)
        else:
            self.keys.sort()    
        self.keys = self.keys[0:int(len(self.keys)//(100/percent))]
        target_fn_temp = {f[0:idxLine]: os.path.join(path,f) for f 
                  in os.listdir(path) 
                  if f.find(target_substring)>=0}  
        self.target_fn = {key: target_fn_temp.get(data_fn.get(key)) for key in self.keys}
        
    def __len__(self): 
        return len(self.keys)

    # load, transform to torch tensor and add dummy "channels" dimension
    def _torchItem(self,fn):
        data = torch.load(fn)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    
    def __getitem__(self, idx):
        
        fnData = self.keys[idx]
        data = self._torchItem(fnData)
        data.requires_grad = True
        target =self._torchItem(self.target_fn.get(fnData))
        return (data,target)

def saveAsPNG(path,fn,volume,minT=0,maxT=0,zoom=2,percentile=5):
    if not os.path.exists(path):
    	os.makedirs(path)  
    fn = os.path.join(path,fn)
    if minT==maxT:
        minT,maxT =np.nanpercentile(volume,percentile),np.nanpercentile(volume,100-percentile)
    if len(volume.shape)==3:
        # center slice:
        z = volume.shape[0]//2
        volume = volume[z,:,:].squeeze()        
    #resample and save:
    volume = scipy.ndimage.zoom(volume, zoom, order=0)
    volume = np.clip((volume-minT)/(maxT-minT),0,1)*255
    imagePNG = volume.astype(np.uint8)
    imageio.imsave(fn,imagePNG)

def crop(output,pad):
    s = output.shape
    if len(s)==5:
        output = output[:,:,pad[0]:s[2]-pad[0],pad[1]:s[3]-pad[1],
                  pad[2]:s[4]-pad[2]]
        return output
    if len(s)==3:
        output = output[pad[0]:s[0]-pad[0],pad[1]:s[1]-pad[1],
                  pad[2]:s[2]-pad[2]]
        return output
    
# compute statistics:
def rmse(data,target,mask):
    s1 = np.sum((data-target)**2*mask)
    s2 = np.sum(mask)
    rmse = np.sqrt(s1/s2)
    return rmse

def psnr(data,target,mask):
    r = np.percentile(target,99)-np.percentile(target,1)
    psnr = 20 * np.log10(r/(rmse(data,target,mask)))
    return psnr

def ssim(data,target,mask):
    # to implement!
    return 0     

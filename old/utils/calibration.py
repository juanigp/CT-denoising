#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:44:06 2019

@author: felix

1) in-place calibration of entire vertebrae but only using voxels defined by mask
2) micro-structural calibration by groups and over voxels defined by mask 
"""
import numpy as np
import kernels
import os.path,shutil
from scipy import signal
import volumeio
from tqdm import tqdm

###########################################
### density in-place calibration ##########
###########################################
R = 14.5
DELTA = (2,1,1)

def local_bmd(volume,kernel):
    volume[np.isnan(volume)] = 0
    volume = signal.convolve(volume,kernel,'same')
    return volume

def _calibration_values(volume,gt,mask=None):
    if mask is None:
        x = gt.reshape(-1)
        y = volume.reshape(-1)
    else:
        x = gt[mask>0]
        y = volume[mask>0]
    m,b = np.polyfit(y, x, 1)
    return m,b
  
def _process_calib_fn(fn_in,fn_out,gt,mask,kernel):
    vol_itk = volumeio.OpenXML(fn_in,kind='Slices') 
    vol_np = volumeio.sitk.GetArrayFromImage(vol_itk)
    dtype = vol_np.dtype 
    
    vol_np = local_bmd(vol_np,kernel)
    m,b = _calibration_values(vol_np,gt,mask)
    vol_np = volumeio.sitk.GetArrayFromImage(vol_itk)*m+b
    
    vol_itk2 = volumeio.sitk.GetImageFromArray(vol_np.astype(dtype))
    volumeio.CopyMetaData(vol_itk,vol_itk2)
    vol_itk2.SetSpacing(vol_itk.GetSpacing())
    volumeio.SaveXML(vol_itk2,filename=fn_out,kind='Slices')

def _mask(fn_mask,kernel):
    EPS = 0.001
    mask = volumeio.OpenXML(fn_mask,kind='Mask',asNumpy=True)
    mask = np.bitwise_or((mask==1),(mask==5))*1.0
    mask = (signal.convolve(mask,kernel,'same')>(1-EPS))*1.0
    return mask

def _gt(fn_gt,kernel):
    gt = volumeio.OpenXML(fn_gt,kind='Slices',asNumpy=True)
    gt = local_bmd(gt,kernel) 
    return gt

def process_all(mode ='XCT'):
    PATH_DATA = r'C:\Users\Juanig\Desktop\juan\data'
    PATH_VOLUMES = os.path.join(PATH_DATA, 'volumes') 
    PATH_GT = os.path.join(PATH_VOLUMES, 'XCT')
    PATH_SLICES = os.path.join(PATH_VOLUMES, 'raw')
    PATH_MASK = os.path.join(PATH_VOLUMES, 'mask')
    PATH_OUT = os.path.join(PATH_VOLUMES, 'calib')
    kernel = kernels.se3D(delta=DELTA,r=R,se_type='norm',asNumpy=True)
    
    if os.path.exists(PATH_OUT):
        shutil.rmtree(PATH_OUT)
    os.makedirs(PATH_OUT)
    fns = [f for f in os.listdir(PATH_SLICES)]
    patients = {f[:2] for f in fns}
    for pat in patients:
        fn_mask = os.path.join(PATH_MASK,pat+'_VertebraSegMask.xml')
        fn_gt = os.path.join(PATH_GT,pat+'_'+mode+'Slices.xml')
        mask = _mask(fn_mask,kernel)
        gt = _gt(fn_gt,kernel)
        to_process = [f for f in fns if f.find(mode)<0 and f[:2]==pat]
        for fn in tqdm(to_process):
            fn_in = os.path.join(PATH_SLICES,fn)
            fn_out = os.path.join(PATH_OUT,fn)
            _process_calib_fn(fn_in,fn_out,gt,mask,kernel)
     
###########################################
### micro-structure calibration ###########
###########################################
def _avg_std(fns,kernel):
    std = 0
    fn_mask = ''
    for i in tqdm(range(len(fns))):
        volume = volumeio.OpenXML(fns[i][0],kind='Slices',asNumpy=True)
        if fns[i][1]!=fn_mask:
            fn_mask = fns[i][1]
            mask = _mask(fn_mask,kernel) 
        hp = volume-local_bmd(volume,kernel)
        std += (hp[mask==1]).std()
    std /=len(fns)
    return std

def _process_calib_fn_ms(fn_in,fn_out,kernel,factor=1.0):
   vol_itk = volumeio.OpenXML(fn_in,kind='Slices') 
   vol_np = volumeio.sitk.GetArrayFromImage(vol_itk)
   dtype = vol_np.dtype 
    
   lp = local_bmd(vol_np,kernel)
   vol_np = (vol_np-lp)*factor+lp
    
   vol_itk2 = volumeio.sitk.GetImageFromArray(vol_np.astype(dtype))
   volumeio.CopyMetaData(vol_itk,vol_itk2)
   vol_itk2.SetSpacing(vol_itk.GetSpacing())
   volumeio.SaveXML(vol_itk2,filename=fn_out,kind='Slices')
   
# generate list of filenames to process
def fns_setting(setting,path_slices,
                path_out = r'C:\Users\Juanig\Desktop\juan\data\volumes\calib_ms',
                path_mask = r'C:\Users\Juanig\Desktop\juan\data\volumes\mask'):      
    fns_mask = [f for f in os.listdir(path_mask)]
    fns_slice = [f for f in os.listdir(path_slices) if f.find(setting)>=0]
    fns = [(f,g) for f in fns_slice for g in fns_mask]  
    fns = [(os.path.join(path_slices,f[0]),
            os.path.join(path_mask,f[1]),
            os.path.join(path_out,f[0])) 
        for f in fns if f[0][:2]==f[1][:2]]
    fns.sort()
    return fns

# fns = ((in_1,mask_1,out_1),...,(in_n,mask_n,out_n))
def calibrate_group(fns,kernel,sd_target=145):
    sd_group = _avg_std(fns,kernel)
    factor = sd_target / sd_group
    for fn in tqdm(fns):
        _process_calib_fn_ms(fn[0],fn[2],kernel,factor)

def calibrate_groups(settings,target = 'XCT'):
    PATH_DATA = r'C:\Users\Juanig\Desktop\juan\data'
    PATH_VOLUMES = os.path.join(PATH_DATA, 'volumes') 
    PATH_GT = os.path.join(PATH_VOLUMES, 'XCT')
    PATH_SLICES = os.path.join(PATH_VOLUMES, 'raw')
    PATH_OUT = os.path.join(PATH_VOLUMES, 'calib')
    if os.path.exists(PATH_OUT): 
        shutil.rmtree(PATH_OUT)    
    os.makedirs(PATH_OUT)
    
    kernel = kernels.se3D(delta=DELTA,r=R,se_type='norm',asNumpy=True)
    sd_target = _avg_std(fns_setting(target,PATH_GT),kernel)
    for setting in settings:
        fns = fns_setting(setting,PATH_SLICES,PATH_OUT)
        calibrate_group(fns,kernel,sd_target)

if __name__=="__main__":    
    # in-place calibration:
    process_all() 
    
    # micro-structural calibation
    settings = ['100_FBPSiem','100_FBPPhil',
            '250_FBPSiem','250_FBPPhil',
            '360_FBPSiem','360_FBPPhil'] 
    calibrate_groups(settings,target = 'XCT')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:00:29 2018

@author: Felix Thomsen & Juan Pisula
"""
import torch
import numpy as np
import scipy.ndimage.morphology as morph
import os
from tqdm import tqdm 
from utils import volumeio
import shutil

P_SIZE = (19,39,39)
P_STRIDE = (10,20,20)
FOLDER_PATCH = '19x39x39'
P_SUBSTRINGS = ['360_FBPPhil','500FBPMedian','XCT']

PATH_MASK = r'C:\Users\Juan Pisula\Desktop\ct_images\mask'
PATH_SLICES = r'C:\Users\Juan Pisula\Desktop\ct_images\slices'
PATH_PATCH = r'C:\Users\Juan Pisula\Desktop\ct_images\felix_patches'

def box_positions(fnMask):
    imMask = volumeio.OpenXML(fnMask,kind='Mask',asNumpy=True)
    imMask = (imMask>0)*1 
    imErode = imMask
    if P_SIZE[0]>1:
        imErode = morph.binary_erosion(imErode,structure=np.ones((2,1,1)),iterations=P_SIZE[0]-1,origin=(-1,0,0))
    if P_SIZE[1]>1:
        imErode = morph.binary_erosion(imErode,structure=np.ones((1,2,1)),iterations=P_SIZE[1]-1,origin=(0,-1,0))
    if P_SIZE[2]>1:
        imErode = morph.binary_erosion(imErode,structure=np.ones((1,1,2)),iterations=P_SIZE[2]-1,origin=(0,0,-1))
    gridx,gridy,gridz = np.meshgrid((np.arange(imMask.shape[1]))%P_STRIDE[1],
                             (np.arange(imMask.shape[0]))%P_STRIDE[0],
                             (np.arange(imMask.shape[2]))%P_STRIDE[2])
    imErode *= np.logical_and(np.logical_and((gridx==0),(gridy==0)),(gridz==0))
    return np.where(imErode==1)

def box_positions_folder(substring_mask = ''):
    #PATH_MASK = 'data/volumes/mask'    
    positions = []
    fnsMask = [f for f in os.listdir(PATH_MASK) if 
               f.find(substring_mask)>=0 or substring_mask=='']
    fnsMask.sort()
    pbar = tqdm(desc = 'patch positions',total=len(fnsMask),leave=False)
    for fn in fnsMask:
        curPositions = box_positions(os.path.join(PATH_MASK,fn))
        positions.append(curPositions)
        pbar.update(1)
    pbar.close()
    return (positions,fnsMask)

#True if the patch does not contain any NaN value
def is_patch_valid(patch):
    if patch is None:
        return False
    result = not (np.any(np.isnan(patch)))
    return result

def grab_and_store(positions,maskIDs,substring_slices = '',counter=0):     
    #PATH_PATCH = os.path.join('data/patches',FOLDER_PATCH)
    #PATH_SLICES = 'data/volumes/slices'
    n = 0
    for i in range(len(positions)):
        n += len(positions[i][0])
    zFill = np.ceil(np.log10(n+1)).astype(int)
    if not os.path.exists(PATH_PATCH):
        os.makedirs(PATH_PATCH)  
    for pat in range(len(positions)):     
        filenamesSlices = [f for f in os.listdir(PATH_SLICES) 
            if f.find(maskIDs[pat])>=0 and f.find('Slices.xml')>=0
            and (substring_slices == '' or f.find(substring_slices)>=0)]
        pbar = tqdm(desc = maskIDs[pat],total=len(filenamesSlices)*len(positions[pat][0]),leave = False)
        for fn in filenamesSlices:
            imSlices = volumeio.OpenXML(os.path.join(PATH_SLICES,fn),kind='Slices',asNumpy=True)
            fnStore = fn.replace('Slices.xml','.pt')
            for i in range(len(positions[pat][0])):
                patch = imSlices[positions[pat][0][i]:positions[pat][0][i]+P_SIZE[0],
                            positions[pat][1][i]:positions[pat][1][i]+P_SIZE[1],
                            positions[pat][2][i]:positions[pat][2][i]+P_SIZE[2]]
                patchID = i + counter
                if is_patch_valid(patch):
                    newfn = fnStore.replace(maskIDs[pat],'P'+str(patchID).zfill(zFill))
                    newfn = os.path.join(PATH_PATCH,newfn) 
                    torch.save(patch,newfn)
                pbar.update(1)
        pbar.close()
        counter+=len(positions[pat][0])
    return counter

def delete_patches():
    #PATH_PATCH = os.path.join('data/patches',FOLDER_PATCH)
    if os.path.exists(PATH_PATCH):
        shutil.rmtree(PATH_PATCH)

def make_patches(delete_old_patches = True):  
    if delete_old_patches:
        delete_patches()
    positions,fnsMask = box_positions_folder()
    maskIDs = [f[0:2] for f in fnsMask]
    for substring in P_SUBSTRINGS:
        grab_and_store(positions,maskIDs,substring_slices = substring)
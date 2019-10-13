#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:00:29 2018

@author: Felix Thomsen & Juan Pisula
"""
import numpy as np
import scipy.ndimage.morphology as morph
from tqdm import tqdm 
import os,shutil,torch,re,random,imageio
import torch.utils.data
import scipy.ndimage

if __name__=="__main__":
    import volumeio
else:
    import utils.volumeio as volumeio

###############################################################################
## Create patches #############################################################
###############################################################################
P_SIZE = (19,39,39)
P_STRIDE = (10,20,20)

def ABSPATH(relpath):
    return os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),relpath))

PATH_VOL = ABSPATH('../data/volumes')
PATH_PATCH = ABSPATH('../data/patches')
# positions:    
def _box_positions(fnMask):
    mask = volumeio.OpenXML(fnMask,kind='Mask',asNumpy=True)
    #mask = np.bitwise_or((mask==1),(mask==5))*1.0
    mask = (mask > 0) * 1.0
    if P_SIZE[0]>1:
        mask = morph.binary_erosion(mask,structure=np.ones((2,1,1)),
                                       iterations=P_SIZE[0]-1,origin=(-1,0,0))
    if P_SIZE[1]>1:
        mask = morph.binary_erosion(mask,structure=np.ones((1,2,1)),
                                       iterations=P_SIZE[1]-1,origin=(0,-1,0))
    if P_SIZE[2]>1:
        mask = morph.binary_erosion(mask,structure=np.ones((1,1,2)),
                                       iterations=P_SIZE[2]-1,origin=(0,0,-1))
    gridx,gridy,gridz = np.meshgrid((np.arange(mask.shape[1]))%P_STRIDE[1],
                             (np.arange(mask.shape[0]))%P_STRIDE[0],
                             (np.arange(mask.shape[2]))%P_STRIDE[2])
    mask *= np.logical_and(np.logical_and((gridx==0),(gridy==0)),(gridz==0))
    return np.where(mask==1)

def _box_positions_folder(path_vol = PATH_VOL):
    positions = []
    path_mask = os.path.join(path_vol,'mask')
    fns = [f for f in os.listdir(path_mask)] 
    fns.sort()
    pbar = tqdm(desc = 'patch positions',total=len(fns),leave=False)
    for fn in fns:
        curPositions = _box_positions(os.path.join(path_mask,fn))
        positions.append(curPositions)
        pbar.update(1)
    pbar.close()
    return (positions,fns)

#True if the patch does not contain any NaN value
def _is_patch_valid(patch):
    if patch is None:
        return False
    result = not (np.any(np.isnan(patch)))
    return result

def folder_patch():
    return str(P_SIZE[0])+'x'+str(P_SIZE[1])+'x'+str(P_SIZE[2])

# generate patches and store them. 
def _grab_and_store(positions,maskIDs,folder_slices = 'calib',
                   path_slices = PATH_VOL,
                   path_patches = PATH_PATCH):
    n = 0
    for i in range(len(positions)):
        n += len(positions[i][0])
    zFill = np.ceil(np.log10(n+1)).astype(int)
    path_slices = os.path.join(path_slices,folder_slices)
    path_patches = os.path.join(path_patches,folder_slices,folder_patch())
    if os.path.exists(path_patches):
        shutil.rmtree(path_patches)
    os.makedirs(path_patches)  
    counter = 0
    for pat,maskID in enumerate(maskIDs):
        fnSlices = [f for f in os.listdir(path_slices) 
            if f.find(maskID)>=0 and f.find('Slices.xml')>=0]
        pbar = tqdm(desc = folder_slices+': '+maskID,total=len(fnSlices)*len(positions[pat][0]),leave = False)
        for fn in fnSlices:
            imSlices = volumeio.OpenXML(os.path.join(path_slices,fn),kind='Slices',asNumpy=True)
            fnStore = fn.replace('Slices.xml','.pt')
            for i in range(len(positions[pat][0])):
                patch = imSlices[positions[pat][0][i]:positions[pat][0][i]+P_SIZE[0],
                            positions[pat][1][i]:positions[pat][1][i]+P_SIZE[1],
                            positions[pat][2][i]:positions[pat][2][i]+P_SIZE[2]]          
                newfn = fnStore.replace(maskID,'P'+str(i + counter).zfill(zFill)+'_'+maskID)
                newfn = os.path.join(path_patches,newfn) 
                if _is_patch_valid(patch): torch.save(patch,newfn)
                pbar.update(1)
        pbar.close()
        counter+=len(positions[pat][0])
    return counter

# call grab_and_store for different values of folder_slices
def _make_patches(folders_slices = ['XCT','calib','calib_ms'],
                 path_vol= PATH_VOL,path_patches = PATH_PATCH):  
    path_mask = os.path.join(path_vol,'mask')
    #positions,fnsMask = _box_positions_folder(path_mask)
    positions,fnsMask = _box_positions_folder(path_vol)
    maskIDs = [f[0:2] for f in fnsMask]
    for folder_slices in folders_slices:
        _grab_and_store(positions,maskIDs,folder_slices,path_vol,path_patches)
        
if __name__=="__main__":
    _make_patches()

###############################################################################
## Load patches ###############################################################
###############################################################################
    
class Dataset(torch.utils.data.Dataset):
    _type_codes = {'FBPPhil':0.,'FBPSiem':1.}
    def __init__(self, ids = [], data_substring='(100|360)_FBPSiem',
                 path_patches = PATH_PATCH,path_data = 'calib/'+folder_patch(),
                 path_gt = 'XCT/'+folder_patch(),percent=100):   
        if len(ids)==0:
            ids = Dataset.available_ids(path = os.path.join(path_patches,path_gt))
        regexp = r'.*'+data_substring+'.*'
        self._type_codes_inv = {self._type_codes[key]:key for key in self._type_codes.keys()}
        path_data_c = os.path.join(path_patches,path_data)
        path_gt_c = os.path.join(path_patches,path_gt)
        
        fns_gt = os.listdir(path_gt_c)
        idx_line = fns_gt[0].find('_')
        fns_gt = [f for f in fns_gt if f[idx_line+1:idx_line+3] in ids]
        keys_t = [f[0:idx_line] for f in fns_gt]
        
        fns_data = [f for f in os.listdir(path_data_c) if f[idx_line+1:idx_line+3] in ids]
        keys_d = [f[0:idx_line] for f in fns_data if re.match(regexp,f) is not None]
        keys = [key for key in keys_t if key in keys_d]
        if percent>0:
            random.shuffle(keys)
        keys = keys[:int(len(keys)//(100/abs(percent)))]
        keys.sort()
        
        self._target_fn = {f[0:idx_line]:os.path.join(path_gt_c,f) for f in fns_gt 
            if f[0:idx_line] in keys}
        
        # replace one of those with only two repetions yielding 3 always reps.
        listOfFns = [f for f in fns_data if re.match(regexp,f) is not None and f[0:idx_line] in keys]
        lofn = {item[:-4]+str(i)+'.pt' for i in range(1,4) for item in listOfFns}        
        missing = list(lofn-set(listOfFns))
        listOfFns += [item[:-4]+str((int(item[-4])%3)+1)+item[-3:] for item in missing]
        listOfFns.sort()
        
        #unique meta-data:
        self.meta_keys = list({(int(item.split('_')[2]),self._type_codes[item.split('_')[3]]) for item in listOfFns})
        self.meta_keys.sort()
        self._data_fn = {key1:{key2:[] for key2 in self.meta_keys} for key1 in keys}
        for f in listOfFns:
            subs = f.split('_')
            patch = subs[0]
            key2 = (int(subs[2]),self._type_codes[subs[3]])
            self._data_fn[patch][key2].append(os.path.join(path_data_c,f))
            
    def __len__(self):
        return len(self._target_fn)
  
    def _torchItem(fn):
        data = torch.load(fn)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)
  
    # key1 = patch number
    def __getitem__(self, idx):
        patch = list(self._data_fn.keys())[idx]
        target = Dataset._torchItem(self._target_fn[patch])       
        reps = len(self._data_fn[patch][self.meta_keys[0]])
        sets = len(self.meta_keys)
        data = torch.zeros((sets*reps,*target.size()[1:]),dtype=torch.float32)
        meta_data = torch.zeros((sets*reps,2))
        for _set in range(sets):
            for _rep in range(reps):
                i = _rep+_set*reps
                data[i,:,:,:] = Dataset._torchItem(self._data_fn[patch][self.meta_keys[_set]][_rep])
                meta_data[i,:] = torch.tensor(self.meta_keys[_set])
        return (data,target,meta_data)

    def available_ids(path = PATH_PATCH+'/XCT/'+folder_patch()):
        fns = os.listdir(path)
        idx_line = fns[0].find('_')
        return sorted(list({f[idx_line+1:idx_line+3] for f in fns}))
    
    # key must be numpy array or pair
    def key_str(self,key):
         return str(key[0])+'_'+self._type_codes_inv[key[1]]    

#if __name__== 'main':    
#    train_data = patches.Dataset(ids = ['5a', '5b'], data_substring='(250|360)_FBPSiem')    
#    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=2, shuffle=True,drop_last = True)
#    train_iter = iter(train_loader)
#    (data,target,metadata) = train_iter.next()    
    
###############################################################################
## read subvolumes - write slices  ############################################
###############################################################################
def fn_list_volumes(path=PATH_VOL+'/calib',data_substring='(100|360)_FBPSiem',ids = ['V8']):
    regexp = r'.*'+data_substring+'.*'
    fns = os.listdir(path)
    idx_line = fns[0].find('_')
    keys = list({'_'.join(f.split('_')[:-1]) for f in fns 
                 if re.match(regexp,f) is not None and f[0:idx_line] in ids})
    fn_list = []
    for _key in keys:
        fn_list.append(sorted([fn for fn in fns if fn[:len(_key)]==_key])[0])
    fn_list.sort()
    return (path,fn_list)
    
def read_subvolume(fn,path=PATH_VOL+'/calib',size = (9,-1,-1)):
    vol = volumeio.OpenXML(os.path.join(path,fn),kind='Slices',asNumpy=True)
    s = np.array(size)
    shape = np.array(vol.shape)
    s[s<0] = shape[s<0]
    #center:
    x = shape//2-s//2
    vol = vol[x[0]:x[0]+s[0],x[1]:x[1]+s[1],x[2]:x[2]+s[2]]
    # remove maybe nans:
    vol[np.isnan(vol)]=0
    # meta-data:
    try:
        l = fn.split('_')
        metadata = torch.tensor([int(l[1]),Dataset._type_codes[l[2]]])
    except: # ground truth case
        metadata = (0,0)
        
    return (vol,metadata)

def saveAsPNG(fn,path,volume,t=(0,0),zoom = 1,percentile=5):    
    minT = t[0]
    maxT = t[1]
    if not os.path.exists(path):
    	os.makedirs(path)  
    fn = os.path.join(path,fn)
    if type(volume) == torch.Tensor:
        volume = volume[0,0,:,:,:].cpu().numpy()
    if minT==maxT:
        minT,maxT =np.nanpercentile(volume,percentile),np.nanpercentile(volume,100-percentile)
    if len(volume.shape)==3:
        # center slice:
        z = volume.shape[0]//2
        volume = volume[z,:,:].squeeze()        
    # save:
    vol2 = scipy.ndimage.zoom(volume, zoom, order=0)
    vol2 = np.clip((volume-minT)/(maxT-minT),0,1)*255
    imagePNG = vol2.astype(np.uint8)
    imageio.imsave(fn,imagePNG)
    return (minT,maxT)

#if __name__ == 'main':
#    import matplotlib.pyplot as plt
#    (path,fn_list) = fn_list_volumes()
#    vol,md = read_subvolume(fn_list[0],path)
#    print(md)
#    plt.figure(1)
#    plt.imshow(vol[4,:,:])
#
#    (path,fn_list) = fn_list_volumes(path= '../data/volumes/XCT',data_substring='XCT')
#    vol,md = read_subvolume(fn_list[0],path)
#    plt.figure(2)
#    plt.imshow(vol[4,:,:])

###############################################################################
## other functions ############################################################
###############################################################################
def adjust_size(patch1,patch2,return_both=True):
    si1 = patch1.size()[2:]
    si2 = patch2.size()[2:]
    offset = (torch.tensor(si1)-torch.tensor(si2))//2
    o1 = torch.clamp(offset,0)
    patch1 = patch1[:,:,o1[0]:si1[0]-o1[0],o1[1]:si1[1]-o1[1],o1[2]:si1[2]-o1[2]]
    if return_both:
        o2 = torch.clamp(-offset,0)
        patch2 = patch2[:,:,o2[0]:si2[0]-o2[0],o2[1]:si2[1]-o2[1],o2[2]:si2[2]-o2[2]]
        return (patch1,patch2)
    else:
        return patch1
    



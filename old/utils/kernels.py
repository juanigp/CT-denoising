#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:30:30 2019

@author: felix
"""
import numpy as np
import torch
import scipy.integrate as integrate
import os.path
  
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'_kernels')

def _element3D(p0,p1,f=lambda r:1,subdiv=100):
    positions = np.meshgrid(np.linspace(p0[1],p1[1],subdiv+1),\
                            np.linspace(p0[0],p1[0],subdiv+1),
                            np.linspace(p0[2],p1[2],subdiv+1))
    dist = (positions[0]**2+positions[1]**2+positions[2]**2)**0.5
    evals= np.array([f(xi)*(xi<=1) for xi in dist])
    average = (evals[1:,1:,1:]+evals[:-1,:-1,:-1])/2
    el = np.mean(average) * np.prod(p1-p0)
    # error only over non-linear function f:
    dif = (evals[:-1,:-1,:-1]-evals[1:,1:,1:])
    error = np.mean(np.abs(dif))*np.prod(p1-p0) 
    return el,error 

# r is between 0 and 1 
# size and anisotropy of element is defined by delta
def _se3D(delta = np.array((1,1,1)),f=lambda r:1,subdiv=100):
    positions = np.meshgrid(np.arange(-delta[1]/2,1+delta[1],delta[1]),\
                            np.arange(-delta[0]/2,1+delta[0],delta[0]),\
                            np.arange(-delta[2]/2,1+delta[2],delta[2]))
    positions = np.maximum(0,positions)
    p0 = positions[:,:-1,:-1,:-1]
    p1 = positions[:,1:,1:,1:]   
    
    rMin = np.sqrt(p0[0]**2+p0[1]**2+p0[2]**2)
    #g=lambda x:1
    elements = np.zeros(p0[0].shape)
    errors = np.zeros(p0[0].shape)
    for x in range(elements.shape[0]):
        for y in range(elements.shape[1]):
            for z in range(elements.shape[2]):
                if rMin[x,y,z]<=1:
                    elements[x,y,z],errors[x,y,z] = _element3D(p0[:,x,y,z],p1[:,x,y,z],f,subdiv)              
    size = np.array(elements.shape)
    se = np.zeros(size*2-1)
    
    sumError = integrate.quad(lambda rr: (4*np.pi*rr**2)*f(rr),0,1)[0]/8-np.sum(elements)
    fac = sumError/np.sum(errors)
    elements += errors*fac
    # copy quadrants to entire space:    
    se[size[0]-1:,size[1]-1:,size[2]-1:] = elements
    se[size[0]-1::-1,size[1]-1:,size[2]-1:] += elements
    se[:,size[1]-1::-1,:] += se[:,size[1]-1:,:]
    se[:,:,size[1]-1::-1] += se[:,:,size[1]-1:]
    return se 

def se3D(delta = (1,1,1),r=1,se_type='const',subdiv=100,asNumpy=False):
    if r==0:
        return torch.ones((1,1,1))*1.0
    delta = np.array(delta)
    delta = delta/r
    sigma2 = 0.45**2
    mask_funcs = {'const':lambda rr:1.0,
            'norm':lambda rr: np.exp(-0.5*((rr)**2)/sigma2)-np.exp(-0.5/sigma2)} 
    acc = 8
    fn = se_type+"_"+str(delta[0])[:acc]+"_"+str(delta[1])[:acc]+"_"+str(delta[2])[:acc]+".pt"
    path = os.path.join(PATH,fn)
    #search here:
    if not os.path.exists(path):
        k = _se3D(delta=delta,f=mask_funcs[se_type],subdiv=subdiv)
        # normalize:
        k /= k.sum()
        torch.save(k,path)
    else:
        k = torch.load(path)
    if asNumpy: return k
    else: return torch.FloatTensor(k)
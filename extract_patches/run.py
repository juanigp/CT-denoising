#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:52:28 2018

@author: felix
"""
#import os
from modules import CNN
from modules import patches
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train neuronal net')
    parser.add_argument('--load',type=str, default=None, const='', nargs='?', help='continue with specific net')
    parser.add_argument('--patches',type=str, default=None, const='replace', nargs='?', help='make patches')
    
    args = parser.parse_args()    
    
    cnn= CNN.CNN()
    if not args.load==None:
       if not (args.load == ''):
           cnn._write_path = args.load
       cnn.nnread()
    
    # create net:
    if not args.patches==None:
        patches.make_patches(delete_old_patches = (args.patches=='replace'))
    
    cnn.train()    
    
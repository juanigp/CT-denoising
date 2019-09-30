# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:22:51 2017

Read and Write of Structural Insight XML to/from itk image
!! -> conversion between numpy and itk-volume switches x and z !!
in numpy, an array is indexed in the opposite order (z,y,x).
z = 0
slice = sitk.GetArrayFromImage(image)[z,:,:]
imshow(slice)
 
@author: Felix Thomsen
"""
from tkinter import Tk
from tkinter import filedialog
import numpy as np
import xml.etree.ElementTree as ET
import datetime as dt
import SimpleITK as sitk
import os.path

from os import listdir

def _getFilename(kind='Gen',openfn=True,addtitle=''):
    root = Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
    root.deiconify()
    root.lift()
    root.focus_force()
    options = {}
    options['defaultextension'] = '.xml'
    if kind=='Gen':
        options['filetypes'] = [('XML files','*.xml'),('All files', '.*')]
        options['title'] = 'Open Structural Insight volume' + addtitle
    if kind=='Slices':
        if openfn:
            options['filetypes'] = [('XML files','*Slices.xml'),('All files', '.*')]
            options['title'] = 'Open Structural Insight Slices'+ addtitle
        else:
            options['filetypes'] = [('XML files','*Slices.xml')]
            options['title'] = 'Save Structural Insight Slices'+ addtitle
    if kind=='Mask':
        if openfn:
            options['filetypes'] = [('XML files','*Mask.xml'),('All files', '.*')]
            options['title'] = 'Open Structural Insight Mask'+ addtitle
        else:
            options['filetypes'] = [('XML files','*Mask.xml')]
            options['title'] = 'Save Structural Insight Mask'+ addtitle
    if openfn:
        fname = filedialog.askopenfilename(parent=root,**options)
    else:
        fname = filedialog.asksaveasfilename(parent=root,**options)
    root.destroy()
    return fname

def _readHeader(filename,enc='latin_1'):
    header = ''
    with open(filename, "rb") as fid:
        headerStart = False
        for line in fid:
            if '<Header>' in line.decode(encoding=enc):
                headerStart = True
            if headerStart:
                header += line.decode(encoding=enc)
                # read only up to end tag since then follows binary data... 
            if '</Header>' in line.decode(encoding=enc):        
                fid.close()
                break
    xml_tree = ET.XML(header)
    return xml_tree

def _readImage(filename,size,bpp):
    # read image
    data = 0
    with open(filename,"rb") as fid:
        fid.seek(-np.prod(size)*bpp,2) 
        if bpp==1:
            data = np.fromfile(fid,dtype=np.uint8)
        if bpp==2:
            data = np.fromfile(fid,dtype=np.int16)
        if bpp==4:
            data = np.fromfile(fid,dtype=np.float32)
        fid.close()        
    image = np.reshape(data, [size[2],size[1],size[0]])
    return image    

def _toNumpy(itk_volume):
    numpyArray = sitk.GetArrayFromImage(itk_volume)  
    return numpyArray

def OpenXML(filename=0,kind='+Mask',addtitle='',asNumpy=False):
    '''
    Opens/Loads a 3D volume with Structural Insight xml-format:
        returns
        slices or mask for kind='Gen'
        slices for kind='Slices'    
        mask for kind='Mask'
        pair of (slices,mask) for kind = '+Mask'
        An OpenFileDialog is called if filename is 0.
    '''
    if kind!='+Mask':
        vol = _OpenXML(filename,kind,addtitle)
        if asNumpy:
            vol = _toNumpy(vol)
        return vol
    else:
        itk_slices = _OpenXML(filename=filename,kind='Slices',addtitle=addtitle)
        filename = itk_slices.GetMetaData('Filename') 
        fnMask = filename[:-10]+'Mask.xml'
        itk_mask = 0
        if os.path.isfile(fnMask):
            itk_mask = OpenXML(filename = filename[:-10]+'Mask.xml',kind='Mask')
        else:
            print('Didn''t find file '+fnMask)     
        if asNumpy:
            return (_toNumpy(itk_slices),_toNumpy(itk_mask))
        return (itk_slices,itk_mask)



      

def _OpenXML(filename=0,kind='Gen',addtitle=''):
    if filename==0:        
        filename = _getFilename(kind,addtitle=addtitle)
    if len(filename)==0:
        return (0,0,0)
    xml_tree = _readHeader(filename)
    
    imageTag = xml_tree.find('Image')
    size = np.zeros(3,int)
    size[0] = imageTag.find('SizeX').text
    size[1] = imageTag.find('SizeY').text
    size[2] = imageTag.find('SizeZ').text
    bpp = int(imageTag.find('PixelSizeByte').text)
    
    res = np.zeros(3,float)
    res[0] = imageTag.find('SpacingX').text
    res[1] = imageTag.find('SpacingY').text
    res[2] = imageTag.find('SpacingZ').text
    
    # read image
    image = _readImage(filename,size,bpp)
    
    originVox = np.zeros(3,float)
    originVox[0] = imageTag.find('X1').text
    originVox[1] = imageTag.find('Y1').text
    originVox[2] = imageTag.find('Z1').text
    
    # create itk image:
    itk_volume = sitk.GetImageFromArray(image)
    itk_volume.SetSpacing(res)
    itk_volume.SetOrigin(res*originVox)
    
    #metadata:
    # set filename:
    itk_volume.SetMetaData('Filename',filename)
    
    # consume complete tree:
    while len(xml_tree)>0:
        # consume first element 
        root = xml_tree[0]
        if (root.tag!='Image'):    
            for i in range(len(root)):
                element = xml_tree[0][i]
                key = root.tag+'/'+element.tag
                if element.text==None:
                    element.text = ''
                itk_volume.SetMetaData(key,element.text)
        xml_tree.remove(xml_tree[0])          
    return itk_volume


def _writeImage(filename,image):
    image = np.reshape(image,-1)
    # open for append and as binary
    with open(filename,"ab") as fid:
        image.tofile(fid)
        fid.close()
    return

def _addValue(xml_tree,itk_volume,key1,key2,default=None):
    keyComp = key1+'/'+key2
    if keyComp in itk_volume.GetMetaDataKeys():
        newText = itk_volume.GetMetaData(keyComp)
        #if newText=='':
        #    newText=' '
    else:
        newText = default
    if newText!=None:
        # find key1:
        parent = xml_tree.find(key1)
        if parent==None:
            #add parent
            parent = ET.SubElement(xml_tree,key1)
        ET.SubElement(parent,key2).text = newText  
            
    return xml_tree    

def CopyMetaData(itk_Source,itk_Dest,withFilename=False):
    keys = itk_Source.GetMetaDataKeys()
    for i in range(len(keys)):
        itk_Dest.SetMetaData(keys[i],itk_Source.GetMetaData(keys[i]))
    if not withFilename:
        itk_Dest.EraseMetaData('Filename')
    return
            
def _createXMLTree(itk_volume):
    sp = '  '
    actualTime = dt.datetime.now()
    header = ET.Element('Header')
    header.text = '\n'
    # Patient
    header = _addValue(header,itk_volume,'Patient','ID','Python')
    header = _addValue(header,itk_volume,'Patient','DOB',actualTime.strftime('%Y%m%d'))
    header = _addValue(header,itk_volume,'Patient','Study','')
    
    # Scan
    header = _addValue(header,itk_volume,'Scan','ID','Python')
    header = _addValue(header,itk_volume,'Scan','Date',actualTime.strftime('%Y%m%d'))
    header = _addValue(header,itk_volume,'Scan','Time',actualTime.strftime('%H:%M:%S'))
    header = _addValue(header,itk_volume,'Scan','Site')
    header = _addValue(header,itk_volume,'Scan','Exposure')
    header = _addValue(header,itk_volume,'Scan','Kernel')
    header = _addValue(header,itk_volume,'Scan','Voltage')
    header = _addValue(header,itk_volume,'Scan','Tableheight')
    header = _addValue(header,itk_volume,'Scan','ComputedExposure')
    header = _addValue(header,itk_volume,'Scan','Pitch')
    header = _addValue(header,itk_volume,'Scan','Manufacturer')
    header = _addValue(header,itk_volume,'Scan','Scanner')
    header = _addValue(header,itk_volume,'Scan','Modality')
    
    header = _addValue(header,itk_volume,'Segmentation','DateTime')
    header = _addValue(header,itk_volume,'Segmentation','SegmentationMode')
    
    header = _addValue(header,itk_volume,'Calibration','DateTime')
    header = _addValue(header,itk_volume,'Calibration','Phantom')
    header = _addValue(header,itk_volume,'Calibration','RMSE')
    header = _addValue(header,itk_volume,'Calibration','Method')
    
    header = _addValue(header,itk_volume,'Registration','RefFilename')
    header = _addValue(header,itk_volume,'Registration','Parameters')
    header = _addValue(header,itk_volume,'Registration','FixedParameters')
    
    size = np.array(itk_volume.GetSize())
    res = np.array(itk_volume.GetSpacing())
    origin = (np.array(itk_volume.GetOrigin())/res).astype(int)
    image = ET.SubElement(header,'Image')
    ET.SubElement(image,'SizeX').text = str(size[0])
    ET.SubElement(image,'SizeY').text = str(size[1])
    ET.SubElement(image,'SizeZ').text = str(size[2])
    ET.SubElement(image,'X1').text = str(origin[0])
    ET.SubElement(image,'X2').text = str(origin[0]+size[0]-1)
    ET.SubElement(image,'Y1').text = str(origin[1])
    ET.SubElement(image,'Y2').text = str(origin[1]+size[1]-1)
    ET.SubElement(image,'Z1').text = str(origin[2])
    ET.SubElement(image,'Z2').text = str(origin[2]+size[2]-1)
    ET.SubElement(image,'SpacingX').text = str(res[0])
    ET.SubElement(image,'SpacingY').text = str(res[1])
    ET.SubElement(image,'SpacingZ').text = str(res[2])
    
    choices = {0:'1',1:'1',2:'2',3:'2',8:'4'}
    ET.SubElement(image,'PixelSizeByte').text = choices.get(itk_volume.GetPixelID(), '0')
    
    #make indents 
    list0 = header.findall('*')
    for i in range(len(list0)-1):
        list0[i].text = '\n'+sp+sp
        list0[i].tail = '\n'+sp
    list0[-1].text = '\n'+sp+sp
    list0[-1].tail = '\n'   
    
    for i in range(len(list0)):
        list1 = list0[i].findall('*')
       
        for j in range(len(list1)-1):
            list1[j].tail = '\n'+sp+sp
        list1[-1].tail = '\n'+sp
    
    return header

def SaveXMLFromNP(volume,filename=0,kind='Slices',addtitle=''):
    if kind!='+Mask':
        itk_volume=sitk.GetImageFromArray(volume)
    else:
        itk_volume = (0,0)
        itk_volume[0] = sitk.GetImageFromArray(volume[0])
        itk_volume[1] = sitk.GetImageFromArray(volume[1])
        
    return SaveXML(itk_volume,filename,kind,addtitle)

def SaveXML(itk_volume,filename=0,kind='Slices',addtitle=''):
    '''
    Saves/Stores a 3D volume with Structural Insight xml-format:
        returns the filename and stores
        slices or mask for kind='Gen'
        slices for kind='Slices'    
        mask for kind='Mask'
        pair of (slices,mask) if itk_volume is pair
        A SaveFileDialog is called if filename is 0.
    '''
    # is itk_volume a tuple (of slices + mask)?
    if type(itk_volume)==sitk.SimpleITK.Image:
        return _SaveXML(itk_volume,filename,kind,addtitle)
    if type(itk_volume)==tuple:
        filename = _SaveXML(itk_volume[0],filename=filename,kind='Slices',addtitle=addtitle)
        if type(itk_volume[1]) == sitk.SimpleITK.Image:
            filenameMask = _SaveXML(itk_volume[1],filename=filename[:-10]+'Mask.xml',kind='Mask')
        else:
            filenameMask= ''
        return (filename,filenameMask)
        
def _SaveXML(itk_volume,filename=0,kind='Slices',addtitle='',enc='latin_1'):
    if filename==0:        
        filename = _getFilename(kind,False,addtitle)
    
    # create xml_tree from metadata:
    xml_tree = _createXMLTree(itk_volume)
    header1 = '<!DOCTYPE Structural InsightImageV3.1.2>\n'
    header2 = ET.tostring(xml_tree,encoding=enc).decode(enc)
    header2 = '\n'.join(header2.split('\n')[1:])
    header = header1 + header2
    with open(filename, "w") as text_file:
        print(header, file=text_file)
        text_file.close()    
    # write image:
    _writeImage(filename,sitk.GetArrayFromImage(itk_volume))
    return filename

def transformFromXML(filename,formatOut = 'vtk'):
    writer = sitk.ImageFileWriter()
    # open from propietary format:
    itk_slices,itk_mask = OpenXML(filename + 'xml')

    writer.SetFileName(filename + formatOut)
    writer.Execute(itk_slices)
    if itk_mask!=0:
        writer.SetFileName(filename[:-7]+'Mask.' + formatOut)
        writer.Execute(itk_mask)

def transformFolderFromXML(path,formatOut= 'vtk'):
    onlyfiles = [f for f in listdir(path) 
        if os.path.isfile(os.path.join(path, f)) 
        and f.find('Slices.xml')>0]
    for f in onlyfiles:
        transformFromXML(os.path.join(path,f[:-3]),formatOut)
    return len(f)
#transformFolderFromXML('Phantoms-xml/Mask',formatOut='vtk')
#transformFolderFromXML('Phantoms-xml/LowRes',formatOut='vtk')


def addRegistration(itk_volume,tm,ref_itk_volume):
    itk_volume.SetMetaData('Registration/RefFilename',ref_itk_volume.GetMetaData('Filename'))
    itk_volume.SetMetaData('Registration/Parameters',str(tm.GetParameters()))
    itk_volume.SetMetaData('Registration/FixedParameters',str(tm.GetFixedParameters()))

def test():
    slices,mask = OpenXML()
    #SaveXML((slices,mask))
        
#itk_volume = OpenXML(kind='Slices')
#image = sitk.GetArrayFromImage(itk_volume)
#SaveXML(itk_volume)
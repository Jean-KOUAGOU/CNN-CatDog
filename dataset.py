#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 01:39:28 2020

@author: aims
"""
from torch.utils.data import Dataset
import os
from PIL import Image

class Datasets(Dataset): #Inherit Dataset
    def __init__(self, root_dir, transform=None): #_u init_u is some of the basic parameters to initialize this class
        self.root_dir = root_dir   #File directory
        self.transform = transform #Transform
        self.images = os.listdir(self.root_dir)#All files in the directory
    
    def __len__(self):#Returns the size of the entire dataset
        return len(self.images)
    
    def __getitem__(self,index):#Return dataset[index] based on index index index
        image_index = self.images[index]#Get the picture from the index index index
        img_path = os.path.join(self.root_dir, image_index)#Get the path name of the image index ed
        img = Image.open(img_path)# Read the picture
        label = img_path.split('/')[-1].split('.')[0]# The label of the picture is obtained from the path name of the picture, which is divided according to the path name.I'm here "E:\\Python Project\\Pytorch\\dogs-vs-cats\\train\cat.0.jpg", so first split with "\", select the last one as ['cat.0.jpg'], then split with "." and select [cat as the label for the picture
        #sample = {'image':img,'label':label}#Create a dictionary from pictures and labels
        if self.transform:
            img = self.transform(img)#Transform Samples
            
        return img, label #Return this sample
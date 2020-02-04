# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:28:40 2020

@author: Tanmay Thakur
"""
import os
import cv2

from main import *


imgFiles = os.listdir('data/')
for i in imgFiles:
    images = os.listdir('out/%s'%i)
    for j in images:
        if(j != "summary.png"):
            final(cv2.imread("out/"+ i + "/" + j))
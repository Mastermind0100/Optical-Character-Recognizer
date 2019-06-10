#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:40:46 2019

@author: Atharva
"""
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import scipy.fftpack 
import matplotlib.pyplot as plt
import imutils

arr_out = []
arr_result = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

model=load_model('modelwtstype.h5')

def clear(imgBW, radius):               #to accept regions of image within the given range
    imgBWcopy = imgBW.copy()
    _,contours,_ = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    
    contourList = [] 
    
    for idx in np.arange(len(contours)):        
        cnt = contours[idx]        
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]           
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)
            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy


def areaop(imgBW, areaPixels):      # to accept contours with area within the given range

    imgBWcopy = imgBW.copy()
    _,contours,_ = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

def sortcnts(cnts):                 # to sort the contours left to right
	
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][0], reverse=False))
 
	return (cnts)

def test(a,b,c,d,i,imd):                # to predict the character present in the region of interest
    test_image=imd[b:b+d,a:a+c]
    test_image= cv2.copyMakeBorder(test_image,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
    test_image = cv2.resize(test_image,(64,64),interpolation = cv2.INTER_AREA)
    #print(test_image.shape)
    
    fig.add_subplot(1, 9, i)
    plt.imshow(test_image)
    
    cv2.resize(test_image,(64,64))
    test_image=(image.img_to_array(test_image))/255
    test_image=np.expand_dims(test_image, axis = 0)
    result=model.predict(test_image)  
    np.reshape(result, 36)
    
    #print(result)
    maxval = np.amax(result)
    index = np.where(result == maxval)
    #print('\n','Predicted Character:',arr_result[index[1][0]],'\n')
    arr_out.append(arr_result[index[1][0]])

im = cv2.imread('smoke.jpg')
img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

#Code for enhancing the image--------------------------------------------------

rows = img.shape[0]
cols = img.shape[1]

imgLog = np.log1p(np.array(img, dtype="float") / 255)

M = 2*rows + 1
N = 2*cols + 1
sigma = 10
(X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
centerX = np.ceil(N/2)
centerY = np.ceil(M/2)
gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
Hhigh = 1 - Hlow

HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

gamma1 = 0.3
gamma2 = 1.5
Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]

Ihmf = np.expm1(Iout)
Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
Ihmf2 = np.array(255*Ihmf, dtype="uint8")

Ithresh = Ihmf2 < 65
Ithresh = 255*Ithresh.astype("uint8")
Iclear = clear(Ithresh, 5)
Iopen = areaop(Iclear, 150)

#------------------------------------------------------------------------------

ih = 1
fig=plt.figure(figsize=(64,64))


_, contours, _ = cv2.findContours(Iopen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

contours = sortcnts(contours)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
    test(x,y,w,h,ih,im)
    ih = ih+1
    
plt.show()
#cv2.imshow('rg',Iopen)
#cv2.imshow('rg3',im)
cv2.imshow('rg2',Ithresh)
#arr_out = arr_out[::-1]
final = ""
for i in arr_out:
    final = final+i

print('\n',final)

cv2.waitKey(0)
cv2.destroyAllWindows()

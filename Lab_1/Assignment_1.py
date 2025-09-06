# -*- coding: utf-8 -*-
import numpy as np
import cv2

def convolve(image : np.array ,  kernel : np.array) -> np.array:
    kernel = np.flip(kernel)
    ih, iw = image.shape
    kh , kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    img_padded = cv2.copyMakeBorder(
        image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT
    )
    result = np.zeros((ih, iw), dtype=np.float32)
    for i in range(ih):
        for j in range(iw):
            region = img_padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    return result 


def Gaussian_Smoothing_Function(u,v,sigma):
    return (1/(2*np.pi*sigma**2))*np.exp(-(u**2 + v**2) /(sigma**2))


def Gaussian_Sharp_Function(u,v,sigma):
    return (-1/np.pi*sigma**4) * (1 - (u**2 + v**2)/(2*sigma**2)) * np.exp(-(u**2 + v**2)/(2*sigma**2))

def Gaussian_Smoothing_kernel(size , sigma):
    k = size // 2
    kernel = np.zeros((size,size ) , dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u = i - k
            v = j - k 
            kernel[i,j] = Gaussian_Smoothing_Function(u,v,sigma)
    kernel /= np.sum(kernel)
    return kernel 

def Gaussian_Sharpenning_kernel(size , sigma):
    k = size // 2
    kernel = np.zeros((size,size ) , dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u = i - k
            v = j - k 
            kernel[i,j] = Gaussian_Sharp_Function(u,v,sigma)
    kernel = kernel - kernel.mean()  # Normalize to zero mean
    return kernel 


def individual_channel_convolution(image , kernel):
    b, g, r = cv2.split(image)
    b_channel_convolved = convolve(b, kernel)
    blue_channel = np.zeros_like(image)
    blue_channel[:, :, 0] =255
    blue_channel[:,:,1] = 255 - b_channel_convolved 
    blue_channel[:,:,2] = 255 - b_channel_convolved 
    
    g_channel_convolved = convolve(g, kernel)
    green_channel = np.zeros_like(image)
    green_channel[:,:,1] = 255 
    green_channel[:,:,2] = 255 - g_channel_convolved 
    green_channel[:,:,0] = 255 - g_channel_convolved

    r_channel_convolved = convolve(r,kernel)
    red_channel = np.zeros_like(image)
    red_channel[:,:,2] = 255
    red_channel[:,:,0] = 255 - r_channel_convolved
    red_channel[:,:,1] = 255 - r_channel_convolved
    
    smooth_convolved_image = cv2.merge((b_channel_convolved, g_channel_convolved, r_channel_convolved))
    normalized_result = cv2.normalize(smooth_convolved_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    #cv2.imshow('Blue Channel',blue_channel)
    #cv2.imshow('Green Channel',green_channel)
    #cv2.imshow('Red Channel',red_channel)
    
    return normalized_result



image = cv2.imread(r'E:\Image Processing Lab\Lena.jpg')

b = image[:, :, 0]
g = image[:, :, 1]
r = image[:, :, 2]

blue_img = np.zeros_like(image)
blue_img[:, :, 0] = b

green_img = np.zeros_like(image)
green_img[:, :, 1] = g

red_img = np.zeros_like(image)
red_img[:, :, 2] = r


smoothing_kernel = Gaussian_Smoothing_kernel(5, 1.67)
sharpening_kernel = Gaussian_Sharpenning_kernel(7, 1.67)

smoothed_image=individual_channel_convolution(image,smoothing_kernel)
sharpened_image=individual_channel_convolution(image,sharpening_kernel)

#hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#smooth_hsv = individual_channel_convolution(hsv_image, smoothing_kernel)
#sharp_hsv = individual_channel_convolution(hsv_image, sharpening_kernel)
    

#cv2.imshow('Original Image',image)
#cv2.imshow('HSV Image',hsv_image)
#cv2.imshow('Smooth HSV',smooth_hsv)
#cv2.imshow('Sharp HSV',sharp_hsv)
#cv2.imshow('Smoothed Image',smoothed_image)
#cv2.imshow('Sharpened Image',sharpened_image)
#cv2.imshow('Blue Channel',blue_img)
#cv2.imshow('Green Channel',green_img)
#cv2.imshow('Red Channel',red_img)








cv2.waitKey(0)
cv2.destroyAllWindows()

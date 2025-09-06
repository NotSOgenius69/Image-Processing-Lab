import cv2
import numpy as np

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


def LoG_Function(u,v,sigma):
    return (-1/(np.pi*sigma**4)) * (1 - (u**2 + v**2)/(2*sigma**2)) * np.exp(-(u**2 + v**2)/(2*sigma**2))

def LoG_kernel(sigma):

    size=int(9*sigma)
    if size%2==0:
        size+=1

    k = size//2

    kernel = np.zeros((size,size ) , dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u = i - k
            v = j - k 
            kernel[i,j] = LoG_Function(u,v,sigma)
    kernel = kernel - kernel.mean() 
    return kernel 

def zero_crossing(convolved_image):
    zc=np.zeros_like(convolved_image,dtype=np.uint8)
    rows,cols=convolved_image.shape
    
    for i in range (1,rows-1):
        for j in range (1,cols-1):
            neighbors = [convolved_image[i-1, j], convolved_image[i+1, j],
                         convolved_image[i, j-1], convolved_image[i, j+1]]
            for n in neighbors:
                if (convolved_image[i,j]*n)<0:
                    zc[i,j]=1
    
    return zc

def local_variance_map(convolved_image,zc,window_size=3):
    rows,cols=convolved_image.shape
    var_map=np.zeros_like(convolved_image,dtype=np.float32)
    pad=window_size//2

    padded_image=cv2.copyMakeBorder(convolved_image,pad,pad,pad,pad,cv2.BORDER_CONSTANT)
    for i in range (1,rows-1):
        for j in range (1,cols-1):
            if zc[i,j]==1:
                center_i=i+pad
                center_j=j+pad
                window=padded_image[center_i-pad:center_i+pad+1,center_j-pad:center_j+pad+1]
                window_mean=np.mean(window)
                variance=np.mean((window-window_mean)**2)
                var_map[i,j]=variance
    
    return var_map

def thresholding(var_map,Th=35):
    result=np.zeros_like(var_map,dtype=np.uint8)
    result[var_map > Th] = 255
    return result


image=cv2.imread(r'E:\Image Processing Lab\lena.jpg',cv2.IMREAD_GRAYSCALE)
sigma=0.8
kernel=LoG_kernel(sigma)

convolved_image=convolve(image,kernel)
zc=zero_crossing(convolved_image)
var_map=local_variance_map(convolved_image,zc)
result=thresholding(var_map)
#print('zero crossing:\n',zc)
#print('local variance map:\n',var_map)


cv2.imshow('Original Image',image)
cv2.imshow('Convolved Image',convolved_image)
cv2.imshow('After thresholding',result)


cv2.waitKey(0)
cv2.destroyAllWindows()
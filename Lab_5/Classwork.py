import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot(title,image):
    plt.title(title)
    plt.imshow(image,cmap='gray')
    plt.axis('off')

def calc_dist(u,v,notch):
    return np.sqrt((u-notch[0])**2+(v-notch[1])**2)


def calc_HNR(shape,notch,anti_notch,D0k,n):
    M,N=shape
    u,v=np.meshgrid(np.arange(M),np.arange(N))

    Dk=calc_dist(u,v,notch)
    Dk_neg=calc_dist(u,v,anti_notch)

    Dk=np.where(Dk==0,1e-10,Dk)
    Dk_neg=np.where(Dk_neg==0,1e-10,Dk_neg)

    H1=1.0/(1.0+(D0k/Dk)**(2*n))
    H2=1.0/(1.0+(D0k/Dk_neg)**(2*n))

    return H1*H2


def gen_mask(shape,notch_pairs,anti_notch_pairs,D0k,n):
    mask=np.ones(shape,dtype=np.float64)

    for notch,anti_notch in zip(notch_pairs,anti_notch_pairs):
        HNR=calc_HNR(shape,notch,anti_notch,D0k,n)
        mask*=HNR

    return mask


img=cv2.imread('./pnois2.jpeg',0)

ft=np.fft.fft2(img)
ft_shift=np.fft.fftshift(ft)
magnitude=np.abs(ft_shift)
angle=np.angle(ft_shift)

notch_pairs=[(272,256),(261,261)]
center_u,center_v=img.shape[0]//2,img.shape[1]//2
anti_notch_pairs=[]
for notch in notch_pairs:
    anti_notch_u=center_u-(notch[0]-center_u)
    anti_notch_v=center_v-(notch[1]-center_v)
    anti_notch_pairs.append((anti_notch_u,anti_notch_v))

D0k=20
n=2
mask=gen_mask(img.shape,notch_pairs,anti_notch_pairs,D0k,n)

filtered_ft=ft_shift*mask
filtered_ft_shift=np.fft.ifftshift(filtered_ft)
filtered_img=np.fft.ifft2(filtered_ft_shift)
filtered_img=np.real(filtered_img)
filtered_img=cv2.normalize(filtered_img,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)




mag=20*(np.log(magnitude+1))
mag=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
ang=cv2.normalize(angle,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

filtered_mag=20*(np.log(np.abs(filtered_ft)+1))
filtered_mag=cv2.normalize(filtered_mag,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)


plt.figure(figsize=(12,12))
plt.subplot(2,3,1)
plot('Original Image',img)

plt.subplot(2,3,2)
plot('Filtered Image',filtered_img)

plt.subplot(2,3,3)
plot('Magnitude',mag)

plt.subplot(2,3,4)
plot('Filtered Magnitude',filtered_mag)

plt.subplot(2,3,5)
plot('Angle',ang)

plt.show()
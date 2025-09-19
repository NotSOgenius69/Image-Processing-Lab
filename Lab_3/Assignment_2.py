import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
    return 1.0/np.sqrt(2.0*np.pi*(sigma**2))*np.exp(-((x-mu)**2)/(2.0*(sigma**2)))


def calc_pdf_cdf(hist,h,w):
    pdf = hist/(h*w)
    cdf=np.zeros((256,1),dtype=np.float32)
    cdf[0]=pdf[0]
    for i in range(1,256):
        cdf[i]=cdf[i-1]+pdf[i]

    cdf*=255
    cdf=cdf.astype(np.uint8)
    return pdf,cdf

def histogram_matching(image,input_cdf,target_cdf):
    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        diff = np.abs(input_cdf[i] - target_cdf)
        mapping[i] = np.argmin(diff)

    output_image = cv2.LUT(image, mapping)

    return output_image


image = cv2.imread(r'E:\Image Processing Lab\grayscale_landscape.jpg', cv2.IMREAD_GRAYSCALE)
h, w = image.shape
hist_img= cv2.calcHist([image],[0],None,[256],[0,256])
input_pdf,input_cdf = calc_pdf_cdf(hist_img,h,w)


x=np.arange(256, dtype=np.float64)

mu1,sigma1=70,15
mu2,sigma2=180,20

g1=gaussian(x,mu1,sigma1)
g2=gaussian(x,mu2,sigma2)

w1,w2=0.5,0.5

target_pdf=w1*g1+w2*g2
target_hist=target_pdf*h*w
_,target_cdf=calc_pdf_cdf(target_hist,h,w)


output_image = histogram_matching(image,input_cdf,target_cdf)
out_img_hist= cv2.calcHist([output_image],[0],None,[256],[0,256])

plt.figure(figsize=(12,20))

plt.subplot(4,2,1)
plt.title('Input Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(4,2,2)
plt.title('Output Image')
plt.imshow(output_image, cmap='gray')
plt.axis('off')

plt.subplot(4,2,3)
plt.title('Input Image Histogram')
plt.plot(out_img_hist, color='green',label='Output Histogram')
plt.plot(hist_img, color='blue',label='Input Histogram')
plt.legend()

plt.subplot(4,2,4)
plt.title('Target Histogram')
plt.plot(target_hist, color='red')

plt.subplot(4,2,5)
plt.title('Input Image PDF')
plt.plot(input_pdf, color='blue', label='Input PDF')

plt.subplot(4,2,6)
plt.title('Output Image PDF')
plt.plot(target_pdf, color='orange', label='Target PDF')

plt.subplot(4,2,7)
plt.title('Input Image CDF')
plt.plot(input_cdf, color='blue', label='Input CDF')

plt.subplot(4,2,8)
plt.title('Output Image CDF')
plt.plot(target_cdf, color='orange', label='Target CDF')

plt.subplots_adjust(hspace=0.5)
plt.show()

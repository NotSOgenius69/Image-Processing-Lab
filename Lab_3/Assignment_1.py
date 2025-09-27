import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_plot(title,hist_bfr,color_bfr,lablel_bfr,hist_aft,color_aft,lablel_aft):
    plt.title(title)
    plt.plot(hist_aft,color=color_aft,label=lablel_aft)
    plt.plot(hist_bfr,color=color_bfr,label=lablel_bfr)
    plt.legend()

def histo(image):
    b,g,r = cv2.split(image)
    hist_b = cv2.calcHist([b],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
    hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
    return hist_b, hist_g, hist_r

def calc_pdf_cdf(hist,h,w):
    pdf = hist/(h*w)
    cdf=np.zeros((256,1),dtype=np.float32)
    cdf[0]=pdf[0]
    for i in range(1,256):
        cdf[i]=cdf[i-1]+pdf[i]

    cdf*=255
    cdf=cdf.astype(np.uint8)
    return cdf

image = cv2.imread(r'E:\Image Processing Lab\col.jpg')
b,g,r = cv2.split(image)
hist_b, hist_g, hist_r = histo(image)

h,w=b.shape
cdf_b = calc_pdf_cdf(hist_b,h,w)
cdf_g = calc_pdf_cdf(hist_g,h,w)
cdf_r = calc_pdf_cdf(hist_r,h,w)

output_b = cv2.LUT(b, cdf_b)
output_g = cv2.LUT(g, cdf_g)
output_r = cv2.LUT(r, cdf_r)

output_image = cv2.merge((output_b, output_g, output_r))
out_hist_b, out_hist_g, out_hist_r = histo(output_image)

out_cdf_b = calc_pdf_cdf(out_hist_b,h,w)
out_cdf_g = calc_pdf_cdf(out_hist_g,h,w)
out_cdf_r = calc_pdf_cdf(out_hist_r,h,w)

plt.figure(figsize=(12,20))
plt.subplot(4,2,1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4,2,2)
plt.title('Histogram Equalized Image')
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4,2,3)
show_plot('Blue Channel Histogram', hist_b, 'blue', 'Original', out_hist_b, 'cyan', 'Equalized')

plt.subplot(4,2,4)
show_plot('Green Channel Histogram', hist_g, 'green', 'Original', out_hist_g, 'lightgreen', 'Equalized')

plt.subplot(4,2,5)
show_plot('Red Channel Histogram', hist_r, 'red', 'Original', out_hist_r, 'orange', 'Equalized')

plt.subplot(4,2,6)
show_plot('Blue Channel CDF', cdf_b, 'blue', 'Original', out_cdf_b, 'cyan', 'Equalized')

plt.subplot(4,2,7)
show_plot('Green Channel CDF', cdf_g, 'green', 'Original', out_cdf_g, 'lightgreen', 'Equalized')

plt.subplot(4,2,8)
show_plot('Red Channel CDF', cdf_r, 'red', 'Original', out_cdf_r, 'orange', 'Equalized')

plt.subplots_adjust(hspace=0.5)
plt.show()

hsv_img=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(hsv_img)
height,width=v.shape
hist_v=cv2.calcHist([v],[0],None,[256],[0,256])
cdf_v=calc_pdf_cdf(hist_v,height,width)


out_v=cv2.LUT(v,cdf_v)
out_hsv=cv2.merge((h,s,out_v))
out_hist_v=cv2.calcHist([out_v],[0],None,[256],[0,256])
out_cdf_v=calc_pdf_cdf(out_hist_v,height,width)

plt.figure(figsize=(12,16))
plt.subplot(2,2,1)
plt.title('HSV Image')
plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
plt.axis('off')

plt.subplot(2,2,2)
plt.title('Equalized HSV Image')
plt.imshow(cv2.cvtColor(out_hsv, cv2.COLOR_HSV2RGB))
plt.axis('off')

plt.subplot(2,2,3)
show_plot('Value Channel Histogram', hist_v, 'blue', 'Original', out_hist_v, 'cyan', 'Equalized')
plt.subplot(2,2,4)
show_plot('Value Channel CDF', cdf_v, 'blue', 'Original', out_cdf_v, 'cyan', 'Equalized')

plt.show()




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
    op_count=0
    for i in range(ih):
        for j in range(iw):
            region = img_padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
            op_count+=kh*kw
            op_count+=(kh*kw-1)
            
    print("Total Operation:",op_count)
    return result 




  
image = cv2.imread(r'C:\Users\User\Downloads\box.jpg',cv2.IMREAD_GRAYSCALE)

kernel = [[1,2,1],[3,2,1],[4,5,6]]
kernel = np.array(kernel, dtype=np.float32)


convolved_img=convolve(image,kernel)
convolved_result = cv2.normalize(convolved_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

U, S, Vt = np.linalg.svd(kernel)
kx = np.sqrt(S[0]) * U[:, 0]      # vertical vector
ky = np.sqrt(S[0]) * Vt[0, :]     # horizontal vector

kx = kx[:, np.newaxis]  
ky = ky[np.newaxis, :]  


temp=convolve(image,kx)
output=convolve(temp,ky)

approx_convolved_result = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


rank1_approx = np.outer(kx,ky)

print('Kernel',kernel)

print('Rank-1 approx',rank1_approx)

abs_error = np.abs(kernel-rank1_approx)

print('Absolute error',abs_error)


cv2.imshow("Original Image", image)
cv2.imshow("Convolve Image",convolved_result)
cv2.imshow("Approx Convolved Image",output)


cv2.waitKey(0)
cv2.destroyAllWindows()
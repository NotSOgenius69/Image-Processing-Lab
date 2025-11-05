import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot(title,image):
    plt.title(title)
    plt.imshow(image,cmap='gray')
    plt.axis('off')

def calc_dist(u,v,notch):
    return np.sqrt((u-notch[0])**2+(v-notch[1])**2)

# ADD THIS NEW FUNCTION
def detect_notch_points(magnitude_spectrum, threshold_percentile=99.5, min_distance=10):
    """
    Automatically detect notch points (periodic noise) in magnitude spectrum
    
    Parameters:
    - magnitude_spectrum: FFT magnitude spectrum
    - threshold_percentile: Percentile threshold to detect peaks (higher = fewer peaks)
    - min_distance: Minimum distance between detected peaks
    
    Returns:
    - notch_pairs: List of detected notch coordinates (excluding DC component)
    """
    center_u, center_v = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
    
    # Apply log transform for better visualization
    mag_log = np.log(magnitude_spectrum + 1)
    
    # Set DC component to 0 to avoid detecting it
    mag_log[center_u, center_v] = 0
    
    # Find threshold based on percentile
    threshold = np.percentile(mag_log, threshold_percentile)
    
    # Find peaks above threshold
    peaks = mag_log > threshold
    
    # Get coordinates of peaks
    peak_coords = np.argwhere(peaks)
    
    # Filter out peaks too close to center (DC component)
    notch_pairs = []
    for coord in peak_coords:
        u, v = coord[0], coord[1]
        # Skip if too close to center
        dist_from_center = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        if dist_from_center > min_distance:
            notch_pairs.append((u, v))
    
    # Remove duplicates that are too close to each other
    filtered_notches = []
    for i, notch in enumerate(notch_pairs):
        too_close = False
        for existing in filtered_notches:
            dist = np.sqrt((notch[0] - existing[0])**2 + (notch[1] - existing[1])**2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            filtered_notches.append(notch)
    
    return filtered_notches


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


img=cv2.imread('./pnois2.jpeg')
b,g,r=cv2.split(img)

filtered_channels=[]
for channel in [b,g,r]:
    ft=np.fft.fft2(channel)
    ft_shift=np.fft.fftshift(ft)
    magnitude=np.abs(ft_shift)
    angle=np.angle(ft_shift)


    notch_pairs = detect_notch_points(magnitude, threshold_percentile=99.9, min_distance=10)

    # print(f"Detected {len(notch_pairs)} notch points:")
    # for i, notch in enumerate(notch_pairs):
    # print(f"  Notch {i+1}: {notch}")

    center_u, center_v = img.shape[0]//2, img.shape[1]//2
    anti_notch_pairs = []
    for notch in notch_pairs:
        anti_notch_u=center_u-(notch[0]-center_u)
        anti_notch_v=center_v-(notch[1]-center_v)
        anti_notch_pairs.append((anti_notch_u,anti_notch_v))

    D0k=5
    n=2
    mask=gen_mask(channel.shape,notch_pairs,anti_notch_pairs,D0k,n)

    filtered_ft=ft_shift*mask
    filtered_ft_shift=np.fft.ifftshift(filtered_ft)
    filtered_channel=np.fft.ifft2(filtered_ft_shift)
    filtered_channel=np.real(filtered_channel)
    filtered_channel=cv2.normalize(filtered_channel,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

    filtered_channels.append(filtered_channel)


filtered_img=cv2.merge(filtered_channels)

mag=20*(np.log(magnitude+1))
mag=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
ang=cv2.normalize(angle,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

filtered_mag=20*(np.log(np.abs(filtered_ft)+1))
filtered_mag=cv2.normalize(filtered_mag,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

# Visualize detected notches on magnitude spectrum
mag_with_notches = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
for notch in notch_pairs:
    cv2.circle(mag_with_notches, (notch[1], notch[0]), 5, (0, 255, 0), 2)
for anti_notch in anti_notch_pairs:
    cv2.circle(mag_with_notches, (anti_notch[1], anti_notch[0]), 5, (0, 0, 255), 2)

plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plot('Original Image',img)

plt.subplot(2,3,2)
plt.title('Magnitude with Detected Notches')
plt.imshow(mag_with_notches)
plt.axis('off')

plt.subplot(2,3,3)
plot('Mask', (mask*255).astype(np.uint8))

plt.subplot(2,3,4)
plot('Filtered Image',filtered_img)

plt.subplot(2,3,5)
plot('Filtered Magnitude',filtered_mag)

plt.subplot(2,3,6)
plot('Angle',ang)

plt.tight_layout()
plt.show()
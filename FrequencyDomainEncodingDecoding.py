import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
image_path = "E://Image Processing Lab/Project/Clue-3.png"   # input image
key_phase = 1234           # key for phase encoding
key_perm = 5678            # key for permutation

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
M, N = img.shape

# =========================
# FORWARD FOURIER TRANSFORM
# =========================
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)

# =========================
# PHASE ENCODING
# =========================
magnitude = np.abs(F_shifted)
phase = np.angle(F_shifted)

# Generate pseudo-random phase mask using key
np.random.seed(key_phase)
phase_mask = np.random.uniform(0, 2*np.pi, (M, N))

# Encode phase
encoded_phase = (phase + phase_mask) % (2*np.pi)

# Reconstruct FFT with encoded phase
F_phase_encoded = magnitude * np.exp(1j * encoded_phase)

# =========================
# KEYED PERMUTATION
# =========================
coeffs = F_phase_encoded.flatten()

np.random.seed(key_perm)  # key for permutation
perm = np.random.permutation(len(coeffs))
scrambled_coeffs = coeffs[perm]

F_encoded = scrambled_coeffs.reshape(M, N)

# =========================
# INVERSE FOURIER TRANSFORM
# =========================
F_encoded_shifted = np.fft.ifftshift(F_encoded)
img_encoded = np.fft.ifft2(F_encoded_shifted).real
img_encoded = np.clip(img_encoded, 0, 255).astype(np.uint8)



# =========================
# DECODING FUNCTION
# =========================
def decode_image(img_encoded, key_phase, key_perm):
    M, N = img_encoded.shape

    # FFT of encoded image
    F = np.fft.fft2(img_encoded)
    F_shifted = np.fft.fftshift(F)

    # Flatten and invert permutation
    coeffs = F_shifted.flatten()
    np.random.seed(key_perm)
    perm = np.random.permutation(len(coeffs))
    inverse_perm = np.argsort(perm)
    descrambled_coeffs = coeffs[inverse_perm].reshape(M, N)

    # Extract magnitude and phase
    magnitude = np.abs(descrambled_coeffs)
    phase = np.angle(descrambled_coeffs)

    # Remove phase mask
    np.random.seed(key_phase)
    phase_mask = np.random.uniform(0, 2*np.pi, (M, N))
    decoded_phase = (phase - phase_mask) % (2*np.pi)

    # Reconstruct original image
    F_decoded = magnitude * np.exp(1j * decoded_phase)
    F_decoded_shifted = np.fft.ifftshift(F_decoded)
    img_decoded = np.fft.ifft2(F_decoded_shifted).real
    img_decoded = np.clip(img_decoded, 0, 255).astype(np.uint8)
    return img_decoded

# =========================
# TEST DECODING
# =========================
img_decoded=decode_image(img_encoded,key_phase,key_perm)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.title('Encoded')
plt.imshow(img_encoded,cmap="gray")
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Decoded')
plt.imshow(img_decoded,cmap="gray")
plt.axis('off')

plt.show()
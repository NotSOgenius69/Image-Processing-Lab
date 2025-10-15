import cv2
import numpy as np
import os
from glob import glob


def calculate_hu_moments(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments[:4].flatten()

def chi_squared_distance(feature1, feature2):
    distance = 0
    for i in range(len(feature1)):
        if (feature1[i] + feature2[i]) != 0:
            distance += ((feature1[i] - feature2[i]) ** 2) / (feature1[i] + feature2[i])
    return distance * 0.5


images_folder = r'E:\Image Processing Lab\Lab_4\Images'
image_files = glob(os.path.join(images_folder, '*.jpg'))

print("=" * 80)
print("Computing Hu Moments for all 24 images")
print("=" * 80)


feature_vectors = {}
for img_path in image_files:
    img_name = os.path.basename(img_path)
    hu_moments = calculate_hu_moments(img_path)
    feature_vectors[img_name] = hu_moments
    print(f"\n{img_name}:")
    print(f"  Hu Moments (1-4): {hu_moments}")

print("\n" + "=" * 80)
print("Chi-squared Distance Matching")
print("=" * 80)

base_image = 'Gray_Lena.jpg'
query_images = ['rotated_90_img1.jpg', 'translated_img2.jpg', 'rotated_90_lena.jpg']

if base_image in feature_vectors:
    base_features = feature_vectors[base_image]
    print(f"\nBase Image: {base_image}")
    print(f"Base Features: {base_features}")
    
    distances = {}
    for query in query_images:
        if query in feature_vectors:
            query_features = feature_vectors[query]
            distance = chi_squared_distance(base_features, query_features)
            distances[query] = distance
            print(f"\n{query}:")
            print(f"  Features: {query_features}")
            print(f"  Chi-squared Distance from base: {distance:.6f}")
    
    if distances:
        best_match = min(distances, key=distances.get)
        print(f"\n{'*' * 60}")
        print(f"BEST MATCH: {best_match} with distance {distances[best_match]:.6f}")
        print(f"{'*' * 60}")


print("\n" + "=" * 80)
print("5x5 Patch Analysis")
print("=" * 80)

input_image_path = os.path.join(images_folder, 'Gray_Lena.jpg')
input_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

h, w = input_img.shape
center_y, center_x = h // 2, w // 2
patch = input_img[center_y-2:center_y+3, center_x-2:center_x+3]

print(f"\nExtracted 5x5 patch from center of Gray_Lena.jpg:")
print(patch)

# # Save the patch as an image (scaled up for visibility)
# patch_scaled = cv2.resize(patch, (100, 100), interpolation=cv2.INTER_NEAREST)
# patch_path = os.path.join(images_folder, 'patch_5x5.jpg')
# cv2.imwrite(patch_path, patch_scaled)
# print(f"\nPatch saved to: {patch_path}")


M_translate = np.float32([[1, 0, 1], [0, 1, 1]])
patch_translated = cv2.warpAffine(patch, M_translate, (5, 5))

patch_mirrored = cv2.flip(patch, 1)

M_45 = cv2.getRotationMatrix2D((2, 2), 45, 1)
patch_rotated_45 = cv2.warpAffine(patch, M_45, (5, 5))

patch_rotated_90 = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)

patch_half = cv2.resize(patch, (3, 3))
patch_half_resized = cv2.resize(patch_half, (5, 5))


patch_variants = {
    'original': patch,
    'translated': patch_translated,
    'mirrored': patch_mirrored,
    'rotated_45': patch_rotated_45,
    'rotated_90': patch_rotated_90,
    'half_size': patch_half_resized
}

print("\n" + "-" * 60)
print("Feature Vectors (Hu Moments) for Patch Variants:")
print("-" * 60)

patch_features = {}
for variant_name, variant_patch in patch_variants.items():
    moments = cv2.moments(variant_patch.astype(np.float64))
    hu_moments = cv2.HuMoments(moments)
    patch_features[variant_name] = hu_moments[:4].flatten()
    print(f"\n{variant_name}:")
    print(f"  Patch:\n{variant_patch}")
    print(f"  Hu Moments (1-4): {patch_features[variant_name]}")


print("\n" + "-" * 60)
print("Chi-squared Distances from Original Patch:")
print("-" * 60)

base_patch_features = patch_features['original']
patch_distances = {}

for variant_name, features in patch_features.items():
    if variant_name != 'original':
        distance = chi_squared_distance(base_patch_features, features)
        patch_distances[variant_name] = distance
        print(f"{variant_name}: {distance:.6f}")

if patch_distances:
    best_patch_match = min(patch_distances, key=patch_distances.get)
    print(f"\n{'*' * 60}")
    print(f"BEST MATCH for original patch: {best_patch_match}")
    print(f"Distance: {patch_distances[best_patch_match]:.6f}")
    print(f"{'*' * 60}")


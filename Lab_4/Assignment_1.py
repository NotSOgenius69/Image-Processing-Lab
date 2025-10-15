import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt


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


print("\n" + "=" * 80)
print("Generating Visualization of All Images and Feature Vectors")
print("=" * 80)

image_groups = {
    'Lena': ['Gray_Lena.jpg', 'translated_lena.jpg', 'mirrored_lena.jpg', 'rotated_45_lena.jpg', 'rotated_90_lena.jpg', 'half_size_lena.jpg'],
    'Image 1': ['Gray_img1.jpg', 'translated_img1.jpg', 'mirrored_img1.jpg', 'rotated_45_img1.jpg', 'rotated_90_img1.jpg', 'half_size_img1.jpg'],
    'Image 2': ['Gray_img2.jpg', 'translated_img2.jpg', 'mirrored_img2.jpg', 'rotated_45_img2.jpg', 'rotated_90_img2.jpg', 'half_size_img2.jpg'],
    'Image 3': ['Gray_img3.jpg', 'translated_img3.jpg', 'mirrored_img3.jpg', 'rotated_45_img3.jpg', 'rotated_90_img3.jpg', 'half_size_img3.jpg']
}

transformation_labels = ['Original', 'Translated', 'Mirrored', 'Rotated 45°', 'Rotated 90°', 'Half Size']

fig, axes = plt.subplots(4, 6, figsize=(24, 18))


for col_idx, label in enumerate(transformation_labels):
    fig.text((col_idx + 0.5) / 6, 0.96, label, ha='center', va='top', 
             fontsize=7, fontweight='bold')

for row_idx, (group_name, image_list) in enumerate(image_groups.items()):
    for col_idx, img_name in enumerate(image_list):
        ax = axes[row_idx, col_idx]
        
        
        img_path = os.path.join(images_folder, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            
            
            if img_name in feature_vectors:
                features = feature_vectors[img_name]
                
                
                feature_text = f"[{features[0]:.2e}, {features[1]:.2e}, {features[2]:.2e}, {features[3]:.2e}]"
                ax.text(0.5, -0.12, feature_text, ha='center', va='top', 
                       transform=ax.transAxes, fontsize=6.5, family='monospace',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', transform=ax.transAxes)
        
        ax.axis('off')
        
        if col_idx == 0:
            ax.text(-0.2, 0.5, group_name, rotation=90, va='center', ha='right', 
                   fontsize=14, fontweight='bold', transform=ax.transAxes)

plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.02, hspace=0.4, wspace=0.5)
plt.show()


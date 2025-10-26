import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt

def obj_geom(image):
    area=np.count_nonzero(image)
    kernel=np.ones((3,3),np.uint8)
    eroded=cv2.erode(image,kernel,iterations=1)
    border=image-eroded
    perimeter=np.count_nonzero(border)

    contours,_=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mxcntr=max(contours,key=cv2.contourArea)
    (x,y),(MA,ma),angle=cv2.fitEllipse(mxcntr)
    a=max(MA,ma)
    b=min(MA,ma)

    (cx,cy),(width,height),angle=cv2.minAreaRect(mxcntr)
    mxD=sqrt(width**2+height**2)

    return area,perimeter,a,b,mxD

def reg_desc(image):
    area,perimeter,a,b,mxD=obj_geom(image)

    compactness=(perimeter**2)/area
    form_factor=(4*np.pi*area)/(perimeter**2)
    eccentricity=sqrt(1-(b**2)/(a**2))
    roundness=(4*area)/(np.pi*mxD**2)

    feature_vec=[compactness,form_factor,eccentricity,roundness]
    return feature_vec

def calc_dist(test_d,train_f_vectors,dist_func):
    return [dist_func(test_d,train_d) for train_d in train_f_vectors]

def kl_div(p,q):
    p=np.array(p)
    q=np.array(q)
    p=p/sum(p)
    q=q/sum(q)

    return sum(p*np.log(p/q))

def eucledian_dist(p,q):
    p=np.array(p)
    q=np.array(q)

    return sqrt(sum((p-q)**2))

def cosine(p,q):
    p=np.array(p)
    q=np.array(q)

    return sum(p*q)/sqrt(sum(p**2))*sqrt(sum(q**2))


train_images_paths=[
    './TrainNtest/train1.jpg',
    './TrainNtest/train2.jpg',
    './TrainNtest/train3.png',
    './TrainNtest/train4.jpg'
]
test_images_paths=[
    './TrainNtest/obj1.jpg',
    './TrainNtest/obj2.png',
    './TrainNtest/obj3.jpg'
]
train_images=[]
test_images=[]
for path in train_images_paths:
    train_images.append(cv2.imread(path,0))

for path in test_images_paths:
    test_images.append(cv2.imread(path,0))


test_f_vectors=[]
train_f_vectors=[]
for image in train_images:
    train_f_vectors.append(reg_desc(image))

for image in test_images:
    test_f_vectors.append(reg_desc(image))

distance_matrix=[]
for test_d in test_f_vectors:
    distance=calc_dist(test_d,train_f_vectors,kl_div)
    distance_matrix.append(distance)

from tabulate import tabulate
row_headers = [f'Test {i + 1}' for i in range(3)]
col_headers = [f'GT {i + 1}' for i in range(4)]

distances_matrix = np.array(distance_matrix)
print(tabulate(distances_matrix[0:3,0:4], headers=col_headers, showindex=row_headers, tablefmt='grid'))
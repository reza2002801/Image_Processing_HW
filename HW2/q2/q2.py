import numpy as np
import cv2
import random


img=cv2.imread('Greek-ship.jpg').astype(np.float64)
patch=cv2.imread('patch.png').astype(np.float64)

def NCC(img, patch):
    g_tilda=patch-np.mean(patch)
    eye=np.ones((patch.shape[0], patch.shape[1]), dtype='float64')
    im=cv2.filter2D((img-np.mean(img)),-1,
                    (patch-np.mean(patch)))/(np.sqrt(np.sum(g_tilda*g_tilda)*
                                                     cv2.filter2D((img-np.mean(img))*(img-np.mean(img)),-1,eye)))

    return np.abs(im)

def check_dist(d,y):
    for i in range(len(l)):
        if abs(d-l[i])<17:
            p[i].append(y)
            return False
    l.append(d)
    f=[]
    f.append(y)
    p.append(f)
    return True

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg
k=1

s=3*patch.shape[0]//8
h=3*patch.shape[1]//8
th=0.3

size = 0.2
m=[]

while size <= 0.6:

    resized = cv2.resize(patch, (0, 0), fx=size, fy=size - 0.018)

    f = k * NCC(img[:, :, 0], resized[:, :, 0])
    f2 = k * NCC(img[:, :, 1], resized[:, :, 1])
    f3 = k * NCC(img[:, :, 2], resized[:, :, 2])
    a = np.where((f > th +0.15) & (f2 > th) & (f3 > th))
    m += list(zip(*a))
    print(a)
    print(m)
    size += 0.1

l=[]
p=[]
for i in m:
    check_dist(i[1], i[0])



for i in range(len(l)):
    a = random.randint(0, 255)
    b = random.randint(0, 255)
    c = random.randint(0, 255)
    g=int(cal_average(p[i]))
    cv2.rectangle(img, pt1=(l[i] - h, g - s), pt2=(l[i] + h, g + s), color=(a, b, c), thickness=5)

cv2.imwrite('res15.jpg',img)


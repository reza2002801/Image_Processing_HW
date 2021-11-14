import numpy as np
import cv2

import  time
def Make_Three_Part(img):
    rows = np.floor(np.size(img, axis=0) / 3).astype(np.uint64)

    b = img[0:rows, :, 0].astype(np.float64)
    g = img[int(rows):int(2 * rows), :, 0].astype(np.float64)
    r = img[int(2 * rows):int(3 * rows), :, 0].astype(np.float64)

    return b, g, r


def half(b, g, r):
    b = b[::2,::2]
    g = g[::2,::2]
    r = r[::2,::2]
    return b, g, r


def find_transition(b, g, r, n, x_b, y_b, x_r, y_r):  #

    for i in range(n):
        b, g, r = half(b, g, r)

    rows, cols = r.shape
    t_b = np.array([[1, 0, x_b], [0, 1, y_b]],np.float64)
    t_r = np.array([[1, 0, x_r], [0, 1, y_r]],np.float64)
    b = cv2.warpAffine(b, t_b, (cols, rows))
    r = cv2.warpAffine(r, t_r, (cols, rows))

    Range_r=20
    Range_b=20
    eMat_b = np.zeros([Range_b, Range_b])
    eMat_r = np.zeros([Range_r, Range_r])
    for i in range(-int(Range_r/2),int(Range_r/2)):
        for j in range(-int(Range_r/2), int(Range_r/2)):
            t_r = np.array([[1, 0, i], [0, 1, j]], np.float64)
            t2_r = cv2.warpAffine(r, t_r, (cols, rows))
            eMat_r[int(j + Range_r/2), int(i + Range_r/2)] = np.sum(np.abs(t2_r - g))

    for i in range(-int(Range_b/2),int(Range_b/2)):
        for j in range(-int(Range_b/2), int(Range_b/2)):
            t_b = np.array([[1, 0, i], [0, 1, j]],np.float64)
            t2_b = cv2.warpAffine(b, t_b, (cols, rows))
            eMat_b[int(j + Range_b/2), int(i +Range_b/2)] = np.sum(np.abs(t2_b - g))

    valyb, valxb = np.where(eMat_b == eMat_b.min()) - np.array([Range_b/2])
    valyr, valxr= np.where(eMat_r == eMat_r.min()) - np.array([Range_r/2])
    f1=(y_b + valyb)
    f2=(x_b + valxb)
    f3=(y_r + valyr)
    f4=(x_r + valxr)
    return f1, f2, f3, f4


def find_Matching(b, g, r):
    num = 3
    x_r = 0
    y_r = 0
    x_b = 0
    y_b = 0
    while (num > -1):
        x_b *= 2
        y_b *= 2
        x_r *= 2
        y_r *= 2
        y_b, x_b, y_r, x_r = find_transition(b, g, r, num, x_b, y_b, x_r, y_r)
        num = num - 1
    return y_b, x_b, y_r, x_r

thresh2=60
thresh4=60
varThresh=35
varThresh2=50
scale=0.1

def smaller_times(img):
    row,col,h=img.shape
    times=0
    while(row >1000 and col >1000):

        img = img[::2,::2,:]
        row/=2
        col/=2
        times+=1
    return img,times



def Upper_Bound_crop(img2,times):
    row, col,h = img2.shape

    l=np.zeros([col])
    for i in range(col):
        for j in range(int(row * scale)):

            if (
                    (img2[j, i, 0] <= thresh2 and img2[j, i, 1] > thresh2 and img2[j, i, 2] > thresh2) or
                    (img2[j, i, 2] <= thresh2 and img2[j, i, 1] > thresh2 and img2[j, i, 0] > thresh2) or
                    (img2[j, i, 1] <= thresh2 and img2[j, i, 2] > thresh2 and img2[j, i, 0] > thresh2) or
                    (img2[j, i, 0] <= thresh2 and img2[j, i, 1] <= thresh2 and img2[j, i, 2] > thresh2) or
                    (img2[j, i, 2] <= thresh2 and img2[j, i, 1] <= thresh2 and img2[j, i, 0] > thresh2) or
                    (img2[j, i, 1] > thresh2 and img2[j, i, 2] <= thresh2 and img2[j, i, 0] <= thresh2)  or
                    (np.sqrt(np.var(img2[j,i,:]))>varThresh)

            ):
                l[i]+=1

    final=np.average(l)
    final=final*2**(times)

    return final
def Bottom_Bound_crop(img2,times):
    row, col,h = img2.shape

    l=np.zeros([col])
    for i in range(col):
        for j in range(row-1,row-int(row * scale),-1):
            if (
                    (img2[j, i, 0] <= thresh2 and img2[j, i, 1] > thresh2 and img2[j, i, 2] > thresh2) or
                    (img2[j, i, 2] <= thresh2 and img2[j, i, 1] > thresh2 and img2[j, i, 0] > thresh2) or
                    (img2[j, i, 1] <= thresh2 and img2[j, i, 2] > thresh2 and img2[j, i, 0] > thresh2) or
                    (img2[j, i, 0] <= thresh2 and img2[j, i, 1] <= thresh2 and img2[j, i, 2] > thresh2) or
                    (img2[j, i, 2] <= thresh2 and img2[j, i, 1] <= thresh2 and img2[j, i, 0] > thresh2) or
                    (img2[j, i, 1] > thresh2 and img2[j, i, 2] <= thresh2 and img2[j, i, 0] <= thresh2)or
                    (np.sqrt(np.var(img2[j, i, :])) > varThresh)

            ):
                l[i] += 1

    final=np.average(l)

    final=final*2**(times)
    return final
def Right_Bound_crop(img2,times):
    row, col, h = img2.shape

    l=np.zeros([col])
    for i in range(row):
        for j in range(col-1,col-int(col * scale),-1):

            if (
                    (np.average(img2[i,j,:])>240)

            ):
                l[i] += 1

    final=np.average(l)

    final=final*2**(times)
    return final
def Left_Bound_crop(img2,times):
    row, col,h = img2.shape

    l=np.zeros([col])
    for i in range(row):
        for j in range(int(col * scale)):

            if (

                    (np.average(img2[i,j,:])>240)


            ):
                l[i] += 1

    final=np.average(l)

    final=final*2**(times)
    return final
def Right_Bound_crop_2(img2,times):
    row, col, h = img2.shape

    l=np.zeros([col])
    for i in range(row):
        for j in range(col-1,col-int(col * scale),-1):

            if (
                    (img2[i, j, 0] <= thresh4 and img2[i, j, 1] > thresh4 and img2[i, j, 2] > thresh4) or
                    (img2[i, j, 2] <= thresh4 and img2[i, j, 1] > thresh4 and img2[i, j, 0] > thresh4) or
                    (img2[i, j, 1] <= thresh4 and img2[i, j, 2] > thresh4 and img2[i, j, 0] > thresh4) or
                    (img2[i, j, 0] <= thresh4 and img2[i, j, 1] <= thresh4 and img2[i, j, 2] > thresh4) or
                    (img2[i, j, 2] <= thresh4 and img2[i, j, 1] <= thresh4 and img2[i, j, 0] > thresh4) or
                    (img2[i, j, 1] > thresh4 and img2[i, j, 2] <= thresh4 and img2[i, j, 0] <= thresh4) or
                    (img2[i, j, 1] <= thresh4 and img2[i, j, 2] <= thresh4 and img2[i, j, 0] <= thresh4) or
                    (np.sqrt(np.var(img2[i,j, :])) > varThresh2-10)

            ):
                l[i] += 1

    final=np.average(l)

    final=final*2**(times)
    return final
def Left_Bound_crop_2(img2,times):
    row, col,h = img2.shape

    l=np.zeros([col])
    for i in range(row):
        for j in range(int(col * scale)):

            if (

                    (img2[i, j, 0] <= thresh4 and img2[i, j, 1] > thresh4 and img2[i, j, 2] > thresh4) or
                    (img2[i, j, 2] <= thresh4 and img2[i, j, 1] > thresh4 and img2[i, j, 0] > thresh4) or
                    (img2[i, j, 1] <= thresh4 and img2[i, j, 2] > thresh4 and img2[i, j, 0] > thresh4) or
                    (img2[i, j, 0] <= thresh4 and img2[i, j, 1] <= thresh4 and img2[i, j, 2] > thresh4) or
                    (img2[i, j, 2] <= thresh4 and img2[i, j, 1] <= thresh4 and img2[i, j, 0] > thresh4) or
                    (img2[i, j, 1] > thresh4 and img2[i, j, 2] <= thresh4 and img2[i, j, 0] <= thresh4) or
                    (img2[i, j, 1] <= thresh4 and img2[i, j, 2] <= thresh4 and img2[i, j, 0] <= thresh4) or
                    (np.sqrt(np.var(img2[i,j :])) > varThresh2+2)


            ):
                l[i] += 1

    final=np.average(l)

    final=final*2**(times)
    return final
def crop(b):
    row, col,h= b.shape
    img2,times=smaller_times(b)
    final = Upper_Bound_crop(img2, times)
    final1 = Bottom_Bound_crop(img2, times)
    # final3 = Right_Bound_crop(img2, times)
    # final4 = Left_Bound_crop(img2, times)
    img3 = b[int(final):int(row - final1), :,:]
    return img3
def crop_2(b):
    row, col,h= b.shape
    img2,times=smaller_times(b)
    # final = Upper_Bound_crop(img2, times)
    # final1 = Bottom_Bound_crop(img2, times)
    final3 = Right_Bound_crop(img2, times)
    final4 = Left_Bound_crop(img2, times)

    img3 = b[:, int(final4):int(col-final3),:]
    return img3
def crop_3(b):
    row, col,h= b.shape
    img2,times=smaller_times(b)
    # final = Upper_Bound_crop(img2, times)
    # final1 = Bottom_Bound_crop(img2, times)
    final3 = Right_Bound_crop_2(img2, times)
    final4 = Left_Bound_crop_2(img2, times)

    img3 = b[:, int(final4):int(col-final3),:]
    return img3

def c_1(img,y_b, x_b, y_r, x_r):
    row, col, h = img.shape

    if(y_b>=0 and y_r>=0):
        k1=max(y_r,y_b)
        k2=row-1
    elif(y_b>=0 and y_r<0):
        k1=y_b
        k2=int(row+y_r)
    elif (y_b < 0 and y_r >= 0):
        k1 = y_r
        k2 = int(row + y_b)
    elif (y_b < 0 and y_r < 0):
        k1 = 0
        k2 = int(row+min(y_r,y_b))


    if (x_b >= 0 and x_r >= 0):
        k3 = max(x_r, x_b)
        k4 = col - 1
    elif (x_b >= 0 and x_r < 0):
        k3 = x_b
        k4 = int(col + x_r)
    elif (x_b < 0 and x_r >= 0):
        k3 = x_r
        k4 = int(col + y_b)
    elif (x_b < 0 and x_r < 0):
        k3 = 0
        k4 = int(col + min(x_r, x_b))

    img = img[int(k1):int(k2), int(k3):int(k4), :]
    img = crop(img)
    img = crop_2(img)
    img = crop_3(img)
    return img


# master-pnp-prok-00400-00458a
# master-pnp-prok-01800-01833a
# master-pnp-prok-01800-01886a

image = cv2.imread('master-pnp-prok-01800-01886a.tif')
image = np.uint8(image)
f=time.time()

b, g, r = Make_Three_Part(image)

b_edge = cv2.Canny(b.astype(np.uint8),100, 200).astype(np.uint8)
g_edge = cv2.Canny(g.astype(np.uint8),50, 200).astype(np.uint8)
r_edge =cv2.Canny(r.astype(np.uint8),50, 200).astype(np.uint8)


y_b, x_b, y_r, x_r = find_Matching(b_edge, g_edge, r_edge)


rows, cols = r.shape
transition_matrix_b = np.array([[1, 0, x_b], [0, 1, y_b]],np.float32)
b2 = cv2.warpAffine(b, transition_matrix_b, (cols, rows))
transition_matrix_r = np.array([[1, 0, x_r], [0, 1, y_r]],np.float32)
r2 = cv2.warpAffine(r, transition_matrix_r, (cols, rows))
im = np.zeros([rows, cols, 3]).astype(np.uint8)

im[:, :, 0]=b
im[:, :, 1]=g
im[:, :, 2]=r
print("tranition values for res03-Amir :","y_b: ",y_b,"x_b: ", x_b, "y_r: ",y_r, "x_r: ",x_r)
im[:, :, 0] = np.int8(b2)
im[:, :, 1] = np.int8(g)
im[:, :, 2] = np.int8(r2)


im=c_1(im.astype(np.uint8),y_b, x_b, y_r, x_r)
im=cv2.resize(im,(cols,rows))
cv2.imwrite('res03-Amir.jpg', im)


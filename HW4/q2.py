import cv2
import numpy as np
import time
from numpy import where
img_park=cv2.imread('park.jpg')

row,col,h=img_park.shape

img_resized=cv2.resize(img_park.copy(),(int(img_park.shape[1]/15),int(img_park.shape[0]/15)),cv2.INTER_AREA)
img_resized = cv2.cvtColor(img_resized.copy(), cv2.COLOR_BGR2Luv)


def make5D_matrix(img):
    row, col ,h=img.shape
    x_mat=np.zeros(img.shape[:2])
    y_mat = np.zeros(img.shape[:2])
    for i in range(x_mat.shape[1]):
        x_mat[:,i]=int(255*(i/col))
    for j in range(x_mat.shape[0]):
        y_mat[j,:]=int(255*(j/row))
    img5D=np.zeros((x_mat.shape[0],x_mat.shape[1],5))
    img5D[:,:,0]=img[:,:,0].copy()
    img5D[:, :, 1] = img[:, :, 1].copy()
    img5D[:, :, 2] = img[:, :, 2].copy()
    img5D[:, :, 3] = y_mat
    img5D[:, :, 4] = x_mat

    # img5D=cv2.merge((img[:,:,0],img[:,:,1],img[:,:,2],x_mat,y_mat))
    return img5D
def find_average(points,img):
    s=np.zeros([5])
    for i in points:
        s+=img[i[0],i[1]]
    return s/len(points)

fin=np.zeros_like(img_resized)
# ggg=np.zeros(1e8)
def hh(img_resized888):

    img_5D=make5D_matrix(img_resized888)
    row,col,h=img_5D.shape
    cluster_matrix_b=np.zeros((row,col))
    cluster_matrix_g=np.zeros((row,col))
    cluster_matrix_r=np.zeros((row,col))
    for i in range(0,row):
        for j in range(0,col):
            # if(np.linalg.norm(rgb-img_5D[i,j,:3])<15):
            #     center=priv5
            # else:
            center=img_5D[i,j,:]
            f=(img_5D-center)
            t=np.sqrt(f[:, :, 0]*f[:, :, 0]+f[:, :, 1]*f[:, :, 1]+f[:, :, 2]*f[:, :, 2]+f[:, :, 3]*f[:, :, 3]+f[:, :, 4]*f[:, :, 4])
            new_center=np.zeros([5])
            new_center.fill(-7)
            s=where(t<25)
            new_center=(img_5D[s[0],s[1],:]).mean(axis=0)
            print(i)
            jp=0

            while(np.sum((center-new_center)**2)>1):
                center = new_center.copy()

                f = (img_5D - center)
                t = np.sqrt(f[:, :, 0]*f[:, :, 0]+f[:, :, 1]*f[:, :, 1]+f[:, :, 2]*f[:, :, 2]+f[:, :, 3]*f[:, :, 3]+f[:, :, 4]*f[:, :, 4])
                s = where(t < 25)
                new_center = (img_5D[s[0],s[1],:]).mean(axis=0)
                jp+=1

            cluster_matrix_b[i,j]=new_center[0]
            cluster_matrix_g[i, j] = new_center[1]
            cluster_matrix_r[i, j] = new_center[2]
            print(jp)

    img_sss=cv2.merge((cluster_matrix_b,cluster_matrix_g,cluster_matrix_r))

    img_resized2=cv2.resize(img_sss.copy(),(int(img_sss.shape[1]),int(img_sss.shape[0])),cv2.INTER_AREA)
    # cv2.imwrite('clus.jpg',img_resized2)
    # img_resized2= cv2.cvtColor(img_resized2.copy(), cv2.COLOR_LAB2BGR)
    return img_resized2

# for i in range(0,6):
#     for j in range(0,8):
g=hh(img_resized)
fin=g.copy()
final=cv2.cvtColor(fin.astype(np.uint8).copy(), cv2.COLOR_Luv2BGR)
img_resized=cv2.resize(final.copy(),(int(img_park.shape[1]),int(img_park.shape[0])),cv2.INTER_AREA)
cv2.imwrite('res055.jpg',img_resized)


# for i in range(50):
# cv2.imwrite('test.jpg',make5D_matrix(img_resized))


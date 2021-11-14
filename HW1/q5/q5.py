import cv2
import numpy as np
import time


img=cv2.imread('Pink.jpg')
row,col,ch=img.shape
def matrix_operation_way(img):
    img1 = img[:row - 2, :col - 2, :].astype(np.float64)
    img2 = img[:row - 2, 1:col - 1, :].astype(np.float64)
    img3 = img[:row - 2, 2:col, :].astype(np.float64)

    img4 = img[1:row - 1, :col - 2, :].astype(np.float64)
    img5 = img[1:row - 1, 1:col - 1, :].astype(np.float64)
    img6 = img[1:row - 1, 2:col, :].astype(np.float64)

    img7 = img[2:row, :col - 2, :].astype(np.float64)
    img8 = img[2:row, 1:col - 1, :].astype(np.float64)
    img9 = img[2:row, 2:col, :].astype(np.float64)

    img1111 = (img1 + img2 + img3 + img4 + img5 + img6 + img7 + img8 + img9).astype(np.float64)
    img88 = np.round(img1111 / 9)
    # print(img88[:, :, 2])
    return img88

#
def openCv_way(img):
    row,col,g=img.shape
    img88 = cv2.boxFilter(img,-1,(3,3))
    # print(img88[1:row-1,1:col-1,2])
    return img88[1:row-1,1:col-1,:]
def forloop_way(img):
    img2=img.astype(np.int64)
    new_image=img[1:row-1,1:col-1,:]
    print(new_image.shape)
    row1,col1,h=new_image.shape
    for k in range(3):
        for i1 in range(0,row1):
            for j1 in range(0,col1):
                i=i1+1
                j=j1+1
                new_image[i1,j1,k]=np.round((img2[i,j,k].astype(np.int64)+img2[i-1,j,k].astype(np.int64)+
                                             img2[i,j-1,k].astype(np.int64)+img2[i+1,j,k].astype(np.int64)+
                                             img2[i,j+1,k].astype(np.int64)+img2[i-1,j-1,k].astype(np.int64)+
                                             img2[i+1,j+1,k].astype(np.int64)+img2[i-1,j+1,k].astype(np.int64)+
                                             img2[i+1,j-1,k].astype(np.int64)).astype(np.float64)/9)
    # print(new_image[:,:,1])
    # print(new_image[:, :, 23])
    return new_image
f1=time.time()
img1=openCv_way(img)
f2=time.time()
img2=forloop_way(img)
f3=time.time()
img3=matrix_operation_way(img)
f4=time.time()





cv2.imwrite("res07.jpg",img1)
cv2.imwrite("res08.jpg",img2)
cv2.imwrite("res09.jpg",img3)
print("openCv_way: ",f2-f1," seconds")
print("forloop_way: ",f3-f2," seconds")
print("matrix_operation_way: ",f4-f3," seconds")



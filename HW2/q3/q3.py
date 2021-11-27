import numpy as np
import cv2
img=cv2.imread('books.jpg')
#
b1 =np.array([[666, 207],
              [379, 104],
              [317, 284],
              [601, 389]])
b2=np.array([[364, 743],
              [412, 467],
              [205, 428],
              [152, 710]])
b3=np.array([[811, 968],
              [623, 668],
              [422, 794],
              [610, 1098]])

def ma(b1):
    w=0
    h=0
    p = b1[0,:] - b1[1,:]
    h = np.linalg.norm(p)
    p = b1[1,:] - b1[2,:]
    w=np.linalg.norm(p)

    mat, _ = cv2.findHomography(np.array([[0, 0], [0, h - 1],[w - 1, h - 1], [w - 1, 0]]),b1, cv2.RANSAC, 5.0)

    print(mat)

    img2=np.zeros((int(h),int(w),3))


    for k in range(3):
        for i in range(int(w)):
            for j in range((int(h))):
                a= np.transpose(np.array([i,j,1]).astype(np.float32))
                b=np.matmul(mat,a)
                b=b/b[2]
                fl=np.floor((b/b[2]))
                d=b-fl
                fl=fl.astype(np.uint)

                img2[j,i,k]=((1-d[0])*(1-d[1])*img[fl[1],fl[0],k]+
                             (1-d[0])*(d[1])*img[fl[1]+1,fl[0],k]+
                             (d[0])*(1-d[1])*img[fl[1],fl[0]+1,k]+
                             (d[0])*(d[1])*img[fl[1]+1,fl[0]+1,k]).astype(np.uint8)


    row,col,h= img2.shape
    dim=(2*col,2*row)
    print(dim)


    return img2
img1=ma(b1)
img2=ma(b2)
img3=ma(b3)
cv2.imwrite('res16.jpg',img1)
cv2.imwrite('res17.jpg',img2)
cv2.imwrite('res18.jpg',img3)
from email.mime import image

import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('Enhance1.JPG')
alpa=0.095

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hsv[:,:,2] = np.uint8(np.floor(255 * np.log10(1+(img_hsv[:,:,2])*alpa)/np.log10(1+255*alpa)))




hist_v = np.cumsum(np.histogram(img_hsv[:,:,2],256,[0,256])[0])
row,col,l=img.shape
v=img_hsv[:,:,2].copy()
v2=v.copy()

fn_v=np.zeros([256])
for i in range(256):
    fn_v[i] = np.floor(((hist_v[i] ) / (row * col )) * 255)
    v[np.where(img_hsv[:,:,2] == i)] = fn_v[i]

img_hsv[:,:,2]=v

img=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)


plt.plot(np.histogram(v,256,[0,256])[0],'g',np.histogram(v2,256,[0,256])[0],'r')
plt.show()



cv2.imwrite('res01.jpg',img)

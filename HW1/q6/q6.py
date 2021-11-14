import numpy as np
import cv2
import matplotlib.pyplot as plt

Dark = cv2.imread('Dark.jpg')
Pink = cv2.imread('Pink.jpg')

D_g_h = np.cumsum(np.histogram(Dark[:, :, 1], 256, [0, 256])[0])
D_r_h = np.cumsum(np.histogram(Dark[:, :, 2], 256, [0, 256])[0])
D_b_h = np.cumsum(np.histogram(Dark[:, :, 0], 256, [0, 256])[0])

P_g_h =np.floor( np.cumsum(np.histogram(Pink[:, :, 1], 256, [0, 256])[0])*(np.size(Dark) / np.size(Pink)))
P_r_h = np.floor(np.cumsum(np.histogram(Pink[:, :, 2], 256, [0, 256])[0])*(np.size(Dark) / np.size(Pink)))
P_b_h = np.floor(np.cumsum(np.histogram(Pink[:, : , 0], 256, [0, 256])[0])*(np.size(Dark) / np.size(Pink)))

er_b = np.zeros([256])
er_g = np.zeros([256])
er_r = np.zeros([256])
new_pic = np.copy(Dark)
new_b=new_pic[:,:,0].copy()
new_g=new_pic[:,:,1].copy()
new_r=new_pic[:,:,2].copy()


for i in range(255):
    er_b[i]=(np.abs(P_b_h-D_b_h[i])).argmin()
    new_b[np.where(Dark[:,:,0]==i)]=er_b[i].astype(np.uint8)
for i in range(255):
    er_g[i] = (np.abs(P_g_h - D_g_h[i])).argmin()
    new_g[np.where(Dark[:, :, 1] == i)] = er_g[i].astype(np.uint8)
for i in range(255):
    er_r[i] = (np.abs(P_r_h - D_r_h[i])).argmin()
    new_r[np.where(Dark[:, :, 2] == i)] = er_r[i].astype(np.uint8)

new_pic[:,:,0]=new_b
new_pic[:,:,1]=new_g
new_pic[:,:,2]=new_r
new_b_Hist = np.histogram(new_pic[:, :, 0], 256, [0, 256])[0]
new_g_Hist = np.histogram(new_pic[:, :, 1], 256, [0, 256])[0]
new_r_Hist = np.histogram(new_pic[:, :, 2], 256, [0, 256])[0]

cv2.imwrite('res11.jpg', new_pic)
contrast = np.arange(0, 256)
plt.plot(contrast, new_b_Hist,'b',new_g_Hist,'g',new_r_Hist,'r')
plt.savefig('res10.jpg')
plt.show()






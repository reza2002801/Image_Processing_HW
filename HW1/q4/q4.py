import cv2
import numpy as np
img=cv2.imread("Flowers.jpg")

img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

img2=img_hsv[:,:,0].copy()
img3=img_hsv[:,:,0].copy()
# img2=(img3<130)+(img3>179)
img2=(img3>=130)*(img3<=179)

thresh3=np.repeat(img2[:, :, np.newaxis], 3, axis=2)
yel=img_hsv.copy()
yel[:,:,0]=(img_hsv[:,:,0]/179)*10 +25

img_hsv1=np.where(thresh3 != (0, 0, 0), yel, img_hsv)
final=cv2.cvtColor(img_hsv1,cv2.COLOR_HSV2BGR)
kernel2 = np.ones((13, 13), np.float32) / 169
blurred_frame = cv2.filter2D(src=final, ddepth=-1, kernel=kernel2)
frame = np.where(thresh3 != (0, 0, 0), final, blurred_frame)


cv2.imwrite("res04.jpg",frame)
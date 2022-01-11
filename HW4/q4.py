import numpy as np
import cv2
from skimage.segmentation import felzenszwalb
img = cv2.imread('birds.jpg')
row1,col1,h=img.shape
img = cv2.resize(img, (1600, 1200), interpolation=cv2.INTER_AREA)
row,col,h=img.shape
def get_points(event,x,y,f,h):
    if (event == cv2.EVENT_LBUTTONDBLCLK):
        c=(y,x)
        points.append(c)

a=input('choose an option by entering its number:\n'
        '1:using client input by clicking on the birds\n'
        '2:use default points\n')
points = []
if(a=='1'):
    print('double click the picture on the points you want \n '
          'at the end press e button to let the program running')
#arbitrary points on the birds
    cv2.namedWindow("birds",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("birds", get_points)
    while 1:
        cv2.imshow("birds",img)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
    cv2.destroyAllWindows()
elif(a=='2'):
    points=[(731, 772),(784, 391),(829, 92)]


# kernel = 2*np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])/3
img2 = img.copy()
seg_img = felzenszwalb(img2, sigma=0.86, min_size=2, scale=400)
t= np.zeros((row,col))
for i in range(len(points)):
    mask=(seg_img == seg_img[points[i][0], points[i][1]])
    t +=  mask* img[:, :, 0]
a=np.where(t>0)
hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hsv_img[a[0],a[1],0]=27
hsv_img[a[0],a[1],1]=255
final=cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR)
final=cv2.resize(final,(col1,row1),interpolation=cv2.INTER_AREA)
cv2.imwrite('res10.jpg', final)
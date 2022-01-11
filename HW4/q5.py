import numpy as np
import os
import math
import cv2
from matplotlib import pyplot as plt
from skimage import filters

#load and show


img_tasbih = plt.imread('tasbih.jpg')
i=cv2.imread('tasbih.jpg')
def get_points(event,x,y,f,h):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print("ggg")
        c=(y,x)
        points.append(c)

a=input('choose an option by entering its number:\n'
        '1:using client input by clicking on the birds\n'
        '2:use default points\n')
numofpoints = 0
points=[]
if(a=='1'):
    print('double click the picture on the points you want \n '
          'at the end press e button to let the program running')
#arbitrary points on the birds
    cv2.namedWindow("tasbih",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("tasbih", get_points)
    while True:
        cv2.imshow("tasbih",i)
        if cv2.waitKey(1)& 0xFF == ord('e') :
            print('dd')
            break
    cv2.destroyAllWindows()
    numofpoints =len(points)
    points=np.array(points)

elif(a=='2'):
    numofpoints = 100
    s = np.linspace(0, 2 * np.pi, numofpoints)
    x = 449 + 251 * np.cos(s)
    y = 399 + 298 * np.sin(s)

    points = np.transpose(np.array([y, x]))

def find_best_index(points2,move_energy,next_step):
    min_energy = math.inf
    index = 0
    for t in range(the_motion_limit*the_motion_limit):
        if move_energy[numofpoints - 1, t] < min_energy:
            min_energy = move_energy[numofpoints - 1, t]
            index = t
    points2[0] = find_position_linear(points2[numofpoints - 1],index)
    for k in range(numofpoints - 1):
        index = next_step[numofpoints - k - 1, index]
        points2[numofpoints - k - 1] =find_position_linear(points2[numofpoints - k - 2],index)
    return points2
def draw_and_save_curve(img,points,fileName=None):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
    plt.margins(0, 0)
    plt.imshow(img)
    plt.plot(points[:, 1], points[:, 0], '-r')
    plt.plot(points[:, 1], points[:, 0], '*b', markersize=5)
    plt.savefig(fileName, dpi=200)
    plt.close('all')
print(points)
#find gradient
# img_blured=cv2.GaussianBlur(img_tasbih,(3,3),8)

# gradiant+=gradiant_r

# p = filters.gaussian(img_tasbih, 3, multichannel=True)

gradiant = np.zeros(img_tasbih.shape[:2])
gradiant_b=np.sqrt(filters.sobel_h(img_tasbih[:,:, 0]) ** 2 + filters.sobel_v(img_tasbih[:,:, 0]) ** 2)
gradiant_g=np.sqrt(filters.sobel_h(img_tasbih[:,:, 1]) ** 2 + filters.sobel_v(img_tasbih[:,:, 1]) ** 2)
gradiant_r=np.sqrt(filters.sobel_h(img_tasbih[:,:, 2]) ** 2 + filters.sobel_v(img_tasbih[:,:, 2]) ** 2)
gradiant+=gradiant_b
gradiant+=gradiant_g
gradiant+=gradiant_r
g=np.round(255*(gradiant-np.min(gradiant))/np.max(gradiant)).astype(np.uint8)
ret2,th2 = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thr=(ret2*np.max(gradiant)/255+np.min(gradiant))+0.0026

mask=np.where(gradiant > thr,1,0)
gradiant = gradiant * mask
gradiant = filters.gaussian(gradiant, 7)

#make directory for images to mp4
name = 'pics'
try:
    os.makedirs(name)
except:
    print('directory alreagy exist')
the_motion_limit=5
frame_rate=5
iteration_number=200
points2=points.copy()
a=2.2e-05
b=3.1e-06
c=10.0
#
def find_position_linear(p,t):
    return p+(t / the_motion_limit - the_motion_limit//2, t % the_motion_limit - the_motion_limit//2)
for i in range(iteration_number):
    move_energy = np.zeros((numofpoints, the_motion_limit*the_motion_limit))
    next_step = move_energy.copy().astype(np.uint8)
    for j in range(numofpoints):
        mean_diffrence = points2 - np.roll(points2, 1, axis=0)
        mean_diffrence = np.sqrt(mean_diffrence[:, 0] ** 2 + mean_diffrence[:, 1] ** 2).mean(axis=0)
        dif_mean = points2 - (points2).mean(axis=0)
        dif_mean = np.sqrt(dif_mean[:, 0] ** 2 + dif_mean[:, 1] ** 2)
        dif_mean = (dif_mean).mean(axis=0)
        for k in range(the_motion_limit*the_motion_limit):
            current = find_position_linear(points2[j],k)
            t=np.zeros((the_motion_limit*the_motion_limit))
            t.fill(np.Inf)
            for l in range(the_motion_limit*the_motion_limit):
                previous = find_position_linear(points2[j-1],l)
                val = move_energy[j - 1, l]
                val += a * (np.linalg.norm(current - previous) ** 2 - mean_diffrence) ** 2
                t[l]=val
            r=np.min(t)
            r += b * (np.linalg.norm(current - (points2).mean( axis=0)) ** 2 - dif_mean / 2) ** 2 *(
                    1 - np.tanh(10*(gradiant[tuple(current.astype(int))]))) -c * gradiant[tuple(current.astype(int))]
            move_energy[j, k] = r
            next_step[j, k] = t.argmin()
    points2=find_best_index(points2,move_energy,next_step)
    if (i % frame_rate == 0):
        draw_and_save_curve(img_tasbih, points2, str(name)+'/'+str(i)+'.jpg')
draw_and_save_curve(img_tasbih, points2, 'res11.jpg')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
img = cv2.imread(name+ '/0.jpg')
height,width ,j=img.shape
video = cv2.VideoWriter('contour.mp4', fourcc, 1, (width, height))
for i in range(0, iteration_number):
    if(i%frame_rate==0):
        img = cv2.imread(name+'/'+str(i) + '.jpg')
        video.write(img)
cv2.destroyAllWindows()
video.release()




import numpy as np
import cv2
from scipy.spatial import Delaunay
import os

def fill_triangle_array(x_cord1,y_cord1,x_cord2,y_cord2,tri_point_num,num_triangle):
    triangle1 = np.zeros([3, 2, num_triangle]).astype(np.float32)
    triangle2 = np.zeros([3, 2, num_triangle]).astype(np.float32)
    for i in range(num_triangle):
        triangle1[0,0,i]=x_cord1[tri_point_num[i, 0]]
        triangle1[0,1,i]= y_cord1[tri_point_num[i, 0]]
        triangle1[1, 0, i] = x_cord1[tri_point_num[i, 1]]
        triangle1[1, 1, i] = y_cord1[tri_point_num[i, 1]]
        triangle1[2, 0, i] = x_cord1[tri_point_num[i, 2]]
        triangle1[2, 1, i] = y_cord1[tri_point_num[i, 2]]
        triangle2[0, 0, i] = x_cord2[tri_point_num[i, 0]]
        triangle2[0, 1, i] = y_cord2[tri_point_num[i, 0]]
        triangle2[1, 0, i] = x_cord2[tri_point_num[i, 1]]
        triangle2[1, 1, i] = y_cord2[tri_point_num[i, 1]]
        triangle2[2, 0, i] = x_cord2[tri_point_num[i, 2]]
        triangle2[2, 1, i] = y_cord2[tri_point_num[i, 2]]
    return triangle1,triangle2
frame_num = 43
first_img = cv2.imread('res01.jpg')
second_img = cv2.imread('res02.jpg')
row,col,h=second_img.shape
dim = (col, row)
second_img_pts = open("second.txt", "r")
first_img_pts = open("first.txt", "r")
Points = first_img_pts.readlines()
Points2 = second_img_pts.readlines()
number = len(Points)
x_cord1 = np.zeros([number]).astype(np.int16)
y_cord1 = np.zeros([number]).astype(np.int16)
x_cord2 = np.zeros([number]).astype(np.int16)
y_cord2 = np.zeros([number]).astype(np.int16)
for i in range(number):
    d = Points[i].split('   ')
    x_cord1[i] = int(d[0])
    y_cord1[i] = int(d[1])
    c = Points2[i].split('   ')
    x_cord2[i] = int(c[0])
    y_cord2[i] = int(c[1])
points = np.zeros([number, 2])
points[:, 0] = x_cord1
points[:, 1] = y_cord1
tri = Delaunay(points)
tri_point_num = tri.simplices
num_triangle = np.size(tri_point_num, axis=0)
dx = x_cord2 - x_cord1
dy = y_cord2 - y_cord1


mask = np.zeros_like(first_img)
triangle1,triangle2=fill_triangle_array(x_cord1,y_cord1,x_cord2,y_cord2,tri_point_num,num_triangle)

name = 'pics'
try:
    os.makedirs(name)
except:
    print('directory alreagy exist')
cv2.imwrite('pics/1.jpg',first_img)
cv2.imwrite('pics/'+str(frame_num+2)+'.jpg',second_img)
f = np.zeros_like(second_img)
for i in range(1, frame_num + 1):
    temp_x = ((x_cord1 + (i / frame_num) * dx)).astype(np.uint16)
    temp_y = ((y_cord1 + (i / frame_num) * dy)).astype(np.uint16)
    for j in range(num_triangle):
        new_pts = np.float32([[temp_x[tri_point_num[j, 0]], temp_y[tri_point_num[j, 0]]], [temp_x[tri_point_num[j, 1]], temp_y[tri_point_num[j, 1]]],
                              [temp_x[tri_point_num[j, 2]], temp_y[tri_point_num[j, 2]]]])
        transform1 = cv2.getAffineTransform(triangle1[:, :, j], new_pts)
        transform2 = cv2.getAffineTransform(triangle2[:, :, j], new_pts)
        triangle = np.int32(new_pts)
        cv2.fillConvexPoly(mask, triangle, (1, 1, 1))
        s = (f == 0) * mask
        warp1 = cv2.warpAffine(first_img, transform1, dim).astype(np.uint16)
        warp2 = cv2.warpAffine(second_img, transform2, dim).astype(np.uint16)
        t=(warp1 * ((frame_num - i)/ frame_num) + warp2 * (i/ frame_num))*s
        f = f + t
        mask[:, :, :] = 0
    if((i+1)==15 or (i+1)==30):
        if(i==14):
            cv2.imwrite( 'res03.jpg', f)
        if (i == 29):
            cv2.imwrite('res04.jpg', f)
        cv2.imwrite('pics/'+str(i+1)+'.jpg', f)
    else:
        cv2.imwrite('pics/' + str(i + 1) + '.jpg', f)
    f = 0
name2='pics/'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
img = cv2.imread(name2+ '1.jpg')
height,width ,j=img.shape
video = cv2.VideoWriter('morph.mp4', fourcc, 30, (width, height))
for i in range(1, frame_num+3):
    img = cv2.imread(name2+str(i) + '.jpg')
    video.write(img)
for i in range(frame_num+2, 0,-1):
    img = cv2.imread(name2+str(i) + '.jpg')
    video.write(img)
cv2.destroyAllWindows()
video.release()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def cluster(x_cord,y_cord,fx,fy,sx,sy):
    first_cluster = []
    second_cluster = []
    for i in range(x_cord.shape[0]):
        dist_from_1=((x_cord[i]-fx)**2+(y_cord[i]-fy)**2)
        dist_from_2 = ((x_cord[i] - sx) ** 2 + (y_cord[i] - sy) ** 2)
        if(dist_from_1<dist_from_2):
            first_cluster.append(i)
        else:
            second_cluster.append(i)

    return first_cluster,second_cluster

def list2numpy(l,x_cord,y_cord):
    numpy_arr_x=np.zeros([len(l),1])
    for i in range(len(l)):
        numpy_arr_x[i]=x_cord[l[i]]
    numpy_arr_y = np.zeros([len(l), 1])
    for i in range(len(l)):
        numpy_arr_y[i] = y_cord[l[i]]
    return numpy_arr_x,numpy_arr_y
def average_finder(l,x_cord,y_cord):
    sum_x=0
    for i in l:
        sum_x+=x_cord[i]
    sum_y = 0
    for i in l:
        sum_y += y_cord[i]

    avg_x=sum_x/len(l)
    avg_y = sum_y / len(l)
    return avg_x,avg_y
def initialize(x_cord,y_cord):
    rnd1=random.randint(0,len(x_cord))
    rnd2=random.randint(0,len(x_cord))

    first_cluster,second_cluster=cluster(x_cord,y_cord,x_cord[rnd1],x_cord[rnd2],y_cord[rnd2],y_cord[rnd2])
    f=first_cluster.copy()
    s=second_cluster.copy()
    return f,s

p=open('Points.txt','r')
points=p.readlines()
num=int(points[0])


x_cord=np.zeros([num,1])
y_cord=np.zeros([num,1])


for i in range(1,int(num)+1):
    x,y=points[i].split(' ')
    x_cord[i-1]=float(x)
    y_cord[i-1]=float(y)

plt.plot( y_cord, x_cord,'.',color='r')
plt.savefig('res01.jpg')
plt.clf()

for i in range(2):
    f,s=initialize(x_cord,y_cord)

    while True:
        clus_mean_first_x,clus_mean_first_y=average_finder(f,x_cord,y_cord)
        clus_mean_sec_x,clus_mean_sec_y=average_finder(s,x_cord,y_cord)

        f1,s1=cluster(x_cord,y_cord,clus_mean_first_x,clus_mean_first_y,clus_mean_sec_x,clus_mean_sec_y)
        if(f1==f):
            break
        else:
            f=f1
            s=s1


    x_1,y_1=list2numpy(f,x_cord,y_cord)
    x_2,y_2=list2numpy(s,x_cord,y_cord)

    plt.plot( y_1, x_1,'.',color='r')
    plt.plot( y_2, x_2,'.',color='g')
    if(i==1):
        plt.savefig('res02.jpg')
    else:
        plt.savefig('res03.jpg')
    plt.clf()
# //////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////
def initialize2(r):
    rnd1=random.randint(0,len(r))
    rnd2=random.randint(0,len(r))

    first_cluster,second_cluster=cluster2(r,r[rnd1],r[rnd2])
    f=first_cluster.copy()
    s=second_cluster.copy()
    return f,s

def cluster2(r,fr,sr):
    first_cluster = []
    second_cluster = []
    for i in range(r.shape[0]):
        dist_from_1=(r[i]-fr)**2
        dist_from_2 = (r[i] - sr) ** 2
        if(dist_from_1<dist_from_2):
            first_cluster.append(i)
        else:
            second_cluster.append(i)

    return first_cluster,second_cluster

def average_finder2(l,r):
    sum_r=0
    for i in l:
        sum_r+=r[i]

    avg_r=sum_r/len(l)


    return avg_r
r=np.zeros([num,1])
r=x_cord**2+y_cord**2

f,s=initialize2(r)

while True:
    first_mean_r=average_finder2(f,r)
    sec_mean_r=average_finder2(s,r)

    f1,s1=cluster2(r,first_mean_r,sec_mean_r)
    if(f1==f):
        break
    else:
        f=f1
        s=s1

x_1,y_1=list2numpy(f,x_cord,y_cord)
x_2,y_2=list2numpy(s,x_cord,y_cord)

plt.plot( y_1, x_1,'.',color='r')
plt.plot( y_2, x_2,'.',color='g')
plt.savefig('res04.jpg')
plt.clf()


import cv2
import numpy as np
from operator import itemgetter

def intersect(p,i):
    inter_y = (((p[0] - 1800) / np.cos((p[1] - 360) * dTetha) - (i[0] - 1800) / np.cos((i[1] - 360)) * dTetha)) / (
                np.tan((p[1] - 360) * dTetha) - np.tan((i[1] - 360) * dTetha))
    inter_x=((p[0] - 1800) -inter_y*np.sin((p[1] - 360) * dTetha))/np.cos((p[1] - 360) * dTetha)
    return inter_y,inter_x
def show_Lines(lines2,img1):
    for i in lines2:
        if(i[0]>360):
            x1 = 0
            x2 = 1500
            if(np.abs(np.sin((i[1] - 360) * dTetha))<1e-9):
                print("d")
                print(i[0],i[1])
                x1=int((i[0]-1800)/np.cos((i[1] - 360) * dTetha))
                x2 = int((i[0]-1800)/ np.cos((i[1] - 360) * dTetha))
                y1=0
                y2=1500
            else:

                try:
                    y1 = int((i[0]-1800 - np.cos((i[1] - 360) * dTetha) * x1) / np.sin((i[1] - 360) * dTetha))
                    y2 = int((i[0]-1800 - np.cos((i[1] - 360) * dTetha) * x2) / np.sin((i[1] - 360) * dTetha))
                except:
                    print('Error returned')



            cv2.line(img1, (y1, x1), (y2, x2), (0, 0, 255), 2)
        else:
            print(i[0],i[1])
            x1 = 0
            x2 = 1500
            y1 = int((i[0]-1800 - np.cos((i[1] - 360) * dTetha) * x1) / np.sin((i[1] - 360) * dTetha))
            y2 = int((i[0]-1800 - np.cos((i[1] - 360) * dTetha) * x2) / np.sin((i[1] - 360) * dTetha))
            cv2.line(img1, (y1, x1), (y2, x2), (0, 0, 255), 2)
    # for i in range(len(lines2)):
    #     for (rho, theta) in lines2[i]:
    #         x1 = 0
    #         x2 = 1500
    #         y1 = int((rho - np.cos(theta) * x1) / np.sin(theta))
    #         y2 = int((rho - np.cos(theta) * x2) / np.sin(theta))
    #
    #         cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img1
def show_Lines1(lines2,img1):
    for i in range(len(lines2)):
        for (rho, theta) in lines2[i]:
            x1 = 0
            x2 = 1500
            y1 = int((rho - np.cos(theta) * x1) / np.sin(theta))
            y2 = int((rho - np.cos(theta) * x2) / np.sin(theta))

            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img2

def local_finder(lines):
    l=[]
    dict={}
    for i in lines:
        if(check_exist(i,l)[0]):
            dict[check_exist(i,l)[1]].append(i)
        else:
            dict[i]=[]
            dict[i].append(i)
            l.append(i)
    l=find_Average(dict)
    l=final(l)
    l,bi,sbi=finallll(l)
    return l,bi,sbi
def finallll(lines):
    l = []
    dict = {}
    for i in lines:
        if (check_exist_angle(i, l)[0]):
            dict[check_exist_angle(i, l)[1]].append(i)
        else:
            dict[i] = []
            dict[i].append(i)
            l.append(i)

    m = []
    for key, value in dict.items():
        m.append(value)
    m.sort(key=len)
    biggest=m[len(m)-1]
    second_biggest=m[len(m)-2]
    sort_biggest= sorted(biggest,key=itemgetter(0))
    sort_second_biggest= sorted(second_biggest,key=itemgetter(0))
    minus_biggest=minus(sort_biggest)
    minus_second_biggest=minus(sort_second_biggest)
    median_biggest=median(minus_biggest)
    median_second_biggest=median(minus_second_biggest)
    bi=sort_biggest
    sbi=sort_second_biggest

    bi=omiter(sort_biggest,median_biggest,minus_biggest)
    sbi=omiter(sort_second_biggest,median_second_biggest,minus_second_biggest)
    h=[]
    # bi.pop(0)
    # sbi.pop(0)

    for i in bi:
        h.append(i)
    for j in sbi:
        h.append(j)

    return h,bi,sbi

def omiter(l,median,minus_l):
    print(l)
    print(minus_l)

    first=0
    last=0
    for i in range(len(l)//2):
        if(((np.abs(minus_l[i]-median)<5 and i!=0) or (i==0 and np.abs(minus_l[i+1]-median)<7))and not(np.abs(minus_l[i]-median)>5)):
            first=i
            break
    for i in range(len(l)-2,len(l)//2,-1):
        if(((np.abs(minus_l[i]-median)<5 and i!=len(l)-2) or (i==len(l)-2 and np.abs(minus_l[i-1]-median)<7))and not(np.abs(minus_l[i]-median)>5)):
            last=i
            break
    print(first)
    print(last)
    print(l[first:last+2])
    return l[first:last+2]


def median(l):
    return l[len(l)//2]


def minus(l):
    m=[]
    for i in range(len(l)):
        if(i!=len(l)-1):
            m.append(l[i+1][0]-l[i][0])
    return m
def minus2(l):
    m=[]
    for i in range(len(l)):
        if(i!=len(l)-1):
            m.append(l[i+1]-l[i])
    return m
def check_exist_angle(p,l):
    for i in l:
        if((np.abs(i[1]-p[1])<20) ):
            return True,i
    return False,None

def final(lines):
    l = []

    for i in lines:
        if (check_intersect(i, l)):
            None
        else:
            l.append(i)

    return l

def check_intersect(p,l):
    for i in l:
        try:
            # inter_y=(((p[0]-1800)/np.cos((p[1]-360)*dTetha) -(i[0]-1800)/np.cos((i[1]-360))*dTetha))/(np.tan((p[1]-360)*dTetha)-np.tan((i[1]-360)*dTetha))
            # print(inter_y)
            inter_y = intersect22222(p, i)[0]
            cond=np.sqrt((p[0]-i[0])**2+(p[1]-i[1])**2)<80 and inter_y<1800 and inter_y>0
            # if(np.abs(i[0]-p[0])<100 and inter_y<1000 and inter_y>0):
            #     return True

            if(np.sqrt((p[0]-i[0])**2+(p[1]-i[1])**2)<20 or cond):
                return True
        except:
            print('d')
    return False
def intersect22222(p,i):
    x1=0
    x2=1500
    y1 = int((i[0] - 1800 - np.cos((i[1] - 360) * dTetha) * x1) / np.sin((i[1] - 360) * dTetha))
    y2 = int((i[0] - 1800 - np.cos((i[1] - 360) * dTetha) * x2) / np.sin((i[1] - 360) * dTetha))

    x12 = 0
    x22 = 1500
    y12 = int((p[0] - 1800 - np.cos((p[1] - 360) * dTetha) * x1) / np.sin((p[1] - 360) * dTetha))
    y22 = int((p[0] - 1800 - np.cos((p[1] - 360) * dTetha) * x2) / np.sin((p[1] - 360) * dTetha))
    A=(y2-y1)/(x2-x1)
    B=(y22-y12)/(x22-x12)
    if(np.abs(A-B)>1e-10):
        inter_x=(A*x2-B*x22-y2+y22)/(A-B)
        inter_y=A*(inter_x-x2)+y2
    else:
        if(np.abs(A*x2-B*x22-y2+y22)<1e-10):
            inter_x=500
            inter_y=500
        else:

            inter_x=None
            inter_y=None
    return (int(inter_y),int(inter_x))
def check_intersect2(p,l,r):
    c=0
    if (p not in r):
        for i in l:
            try:

                inter_y=(((p[0]-1800)/np.cos((p[1]-360)*dTetha) -(i[0]-1800)/np.cos((i[1]-360))*dTetha))/(np.tan((p[1]-360)*dTetha)-np.tan((i[1]-360)*dTetha))
                print("hhg",p, inter_y,i)
                cond=inter_y<1800 and inter_y>-1500
                # if(np.abs(i[0]-p[0])<100 and inter_y<1000 and inter_y>0):
                #     return True

                if( cond):
                    r.append(c)
                    return True,r
            except:
                print('d')
            c+=1
    return False,None
def find_Average(dict):
    m=[]
    for key, value in dict.items():
        m.append(value)
    l=[]
    for i in m:
        s1=0
        s2=0
        for j in i:
            s1+=j[0]
            s2+=j[1]
        if(len(i)!=0):
            l.append((int(s1/len(i)),s2/len(i)))
    return l

def check_exist(p,l):

    for i in l:

        if((np.abs(i[0]-p[0])<15)):
            return True,i

    return False,None

def make_hough_space(edgesPoint1):
    rho_theta1 = np.zeros((3600, 720))
    for i in edgesPoint1:
        for t in range(720):
            t0 = t - 360
            rho = np.sin(t0 * dTetha) * (i[1]) + np.cos(t0 * dTetha) * (i[0])
            # if(rho<0):
            #     print(rho)
            rho_theta1[1800 + int(rho), t] += 1
    return rho_theta1

img1=cv2.imread('im01.jpg')
img2=cv2.imread('im02.jpg')

dTetha=np.pi/720
tetha=0
rho=0


gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
imgCanny1=cv2.Canny(gray1,50,500)
imgCanny2=cv2.Canny(gray2,50,500)
cv2.imwrite('res01.jpg',imgCanny1)
cv2.imwrite('res02.jpg',imgCanny2)


edgesPoint1=list(zip(np.where(imgCanny1==255)[0],np.where(imgCanny1==255)[1]))
edgesPoint2=list(zip(np.where(imgCanny2==255)[0],np.where(imgCanny2==255)[1]))

rho_theta1=make_hough_space(edgesPoint1)
rho_theta2=make_hough_space(edgesPoint2)

cv2.imwrite('res03_hough_space.jpg',rho_theta1)
cv2.imwrite('res04_hough_space.jpg',rho_theta2)

lines1=list(zip(np.where(rho_theta1>150)[0],np.where(rho_theta1>150)[1]))
lines2=list(zip(np.where(rho_theta2>150)[0],np.where(rho_theta2>150)[1]))

i1=show_Lines(lines1,img1.copy())
i2=show_Lines(lines2,img2.copy())


cv2.imwrite('res05_lines.jpg',i1)
cv2.imwrite('res06_lines.jpg',i2)

#
l,bi,sbi=local_finder(lines1)
f1=show_Lines(bi,img1.copy())
f1=show_Lines(sbi,f1.copy())

l2,bi2,sbi2=local_finder(lines2)
f2=show_Lines(bi2,img2.copy())
f2=show_Lines(sbi2,f2.copy())


cv2.imwrite('res07_chess.jpg',f1)
cv2.imwrite('res08_chess.jpg',f2)
for i in bi:
    for j in sbi:
        cv2.circle(img1,intersect22222(i,j),10,(0,255,0),2)

for i in bi2:
    for j in sbi2:
        cv2.circle(img2,intersect22222(i,j),10,(0,255,0),2)

cv2.imwrite('res09_corners.jpg',img1)
cv2.imwrite('res10_corners.jpg',img2)


import cv2
import numpy as np
import scipy
import skimage.segmentation

img=cv2.imread('slic.jpg')
img_lab=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2Lab)
sobelx_kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobely_kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
x_edge=cv2.filter2D(img_lab[:,:,0],-1,sobelx_kernel).astype(np.float32)
y_edge=cv2.filter2D(img_lab[:,:,0],-1,sobely_kernel).astype(np.float32)
gradiant=x_edge.copy()
gradiant=np.sqrt(x_edge*x_edge+y_edge*y_edge)

position_effect=0.025

def make_centers(img,edge,num_of_points):
    row,col,h=img.shape
    k=int(np.sqrt(num_of_points))

    step_row=row//k
    step_col=col//k
    min_grad_tresh=row // (k * 5)
    x_cord=[]
    y_cord=[]
    # color=[]
    f=0
    for i in range(k):
        for j in range(k):
            f+=1

            y=step_row*i+step_row//2
            x=step_col*j+step_col//2
            min_gradient=np.min(edge[y-min_grad_tresh:y+min_grad_tresh,
                                x-min_grad_tresh:x+min_grad_tresh])
            a=np.where(edge[y-min_grad_tresh:y+min_grad_tresh,
                                x-min_grad_tresh:x+min_grad_tresh]==min_gradient)


            xn=x-min_grad_tresh+a[1][a[0].shape[0]//2]
            yn = y - min_grad_tresh + a[0][a[0].shape[0]//2]

            # cv2.circle(img.copy(),(xn,yn),10,(255,0,0),5)
            # color.append(img[yn,xn,:])
            x_cord.append(xn)
            y_cord.append(yn)

    return x_cord,y_cord
def find_square(x,y,S,row,col):
    return max(x - S, 0),min(x + S, col),max(y - S, 0),min(y + S, row)

def find_delta(x,y,S,row,col,position_effect):
    helper = np.indices((row, col)).reshape(2, row, col)
    left, right, top, bottom = find_square(x, y, S, row, col)

    deltal = (img[top:bottom, left:right, 0] - img[y, x, 0]) ** 2
    deltaa = (img[top:bottom, left:right, 1] - img[y, x, 1]) ** 2
    deltab = (img[top:bottom, left:right, 2] - img[y, x, 2]) ** 2
    deltax = (helper[0, top:bottom, left:right] - y) ** 2
    deltay = (helper[1, top:bottom, left:right] - x) ** 2
    delta_lab = deltal + deltaa + deltab
    delta_xy = deltax + deltay
    total_delta = delta_lab + position_effect * delta_xy
    return total_delta,left, right, top, bottom
def Cluster(img,x_cord,y_cord,S,position_effect):
    row,col,h=img.shape

    # clus = np.zeros((row, col, 2))
    a= np.zeros((row, col))
    b= np.zeros((row, col))
    # clus[:, :, 1] = np.Inf
    la=np.linalg.norm(img)
    b.fill(la)

    for i in range(len(x_cord)):
        # print(i)
        total_delta,left, right, top, bottom=find_delta(x_cord[i],y_cord[i],S,row,col,position_effect)
        mask = np.where(total_delta[:bottom - top, :right - left] < b[top:bottom, left:right],1,0)
        b[top:bottom, left:right] =   (1-mask) * b[top:bottom,left:right]+mask * total_delta[:bottom - top, :right - left]
        mask2 = np.where(total_delta[:bottom - top, :right - left] == b[top:bottom, left:right],1,0)
        a[top:bottom, left:right] = mask2 * i + (1-mask2) * a[top:bottom, left:right]

    # cv2.imwrite('k.jpg',u)

    # for j in range(len(x_cord)):
    #
    #     a=np.where(u==j)
    #     try:
    #         x_av=int(np.mean(a[1]))
    #         y_av = int(np.mean(a[0]))
    #         x_cord[j]=x_av
    #         y_cord[j] = y_av
    #         g=img[a[0],a[1],:]
    #         color[j]=g.mean(axis=0).astype(np.uint8)
    #     except:
    #         print('error')
    #         # print(a)
    return a,x_cord,y_cord
def SLIC(img,img_lab,edge,num_of_points):
    x_cord,y_cord=make_centers(img_lab,edge,num_of_points)
    K=np.sqrt(num_of_points)
    S=int(img.shape[1]//K)

    for i in range(1):
        cl_img,x_cord,y_cord = Cluster(img_lab,x_cord,y_cord,S,position_effect)
        cl_img = scipy.signal.medfilt2d(cl_img, 25)
    res = img.copy()
    mask = skimage.segmentation.find_boundaries(cl_img, mode='thick')
    res[:, :, 0] = mask * 0 + (1-mask) * res[:, :, 0]
    res[:, :, 1] = mask * 0 + (1-mask) * res[:, :, 1]
    res[:, :, 2] = mask * 255 +(1-mask) * res[:, :, 2]
    return res

num_of_points=64
res=SLIC(img,img_lab,gradiant,num_of_points)
cv2.imwrite('res06.jpg', res)
num_of_points=256
res=SLIC(img,img_lab,gradiant,num_of_points)
cv2.imwrite('res07.jpg', res)
num_of_points=1024
res=SLIC(img,img_lab,gradiant,num_of_points)
cv2.imwrite('res08.jpg', res)
num_of_points=2048
res=SLIC(img,img_lab,gradiant,num_of_points)
cv2.imwrite('res09.jpg', res)



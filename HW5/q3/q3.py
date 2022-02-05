import cv2
import numpy as np


img_A=cv2.imread('res08.jpg')
img_B=cv2.imread('res09.jpg')

row,col,h=img_A.shape
mask=np.zeros_like(img_A)
mask[:,col//2:,0]=1
mask=mask[:,:,0].astype(np.float64)

def create_stack(img_2D,iter=5):
  img_2D=img_2D.astype(np.float64)
  laplacian=[]
  gausian=[]
  sigma=1
  p=7
  temp=img_2D.copy()
  for i in range(iter):
    img=cv2.GaussianBlur(img_2D,(p,p),sigmaX=sigma,sigmaY=sigma).astype(np.float64)
    gausian.append(img.copy())
    laplacian.append((temp-img).astype(np.float64))
    temp=img.copy()
    sigma*=6
    p+=8
  return gausian,laplacian

iter=36

mask_gau_stack, mask_lap_stack=create_stack(mask,iter)
A_gau_stack_b,A_lap_stack_b=create_stack(img_A[:,:,0],iter)
A_gau_stack_g,A_lap_stack_g=create_stack(img_A[:,:,1],iter)
A_gau_stack_r,A_lap_stack_r=create_stack(img_A[:,:,2],iter)

B_gau_stack_b,B_lap_stack_b=create_stack(img_B[:,:,0],iter)
B_gau_stack_g,B_lap_stack_g=create_stack(img_B[:,:,1],iter)
B_gau_stack_r,B_lap_stack_r=create_stack(img_B[:,:,2],iter)

def norm(img):
  img-=np.min(img)
  img=np.floor(255*(img/np.max(255))).astype(np.uint8)
  return img

out_put_lap_b=[]
out_put_lap_g=[]
out_put_lap_r=[]
for i in range(iter):
  temp_b=(A_lap_stack_b[i]*mask_gau_stack[i]+(1-mask_gau_stack[i])*B_lap_stack_b[i]).astype(np.float64)
  temp_g=(A_lap_stack_g[i]*mask_gau_stack[i]+(1-mask_gau_stack[i])*B_lap_stack_g[i]).astype(np.float64)
  temp_r=(A_lap_stack_r[i]*mask_gau_stack[i]+(1-mask_gau_stack[i])*B_lap_stack_r[i]).astype(np.float64)
  out_put_lap_b.append(temp_b)
  out_put_lap_g.append(temp_g)
  out_put_lap_r.append(temp_r)

first_b=(A_gau_stack_b[iter-1]*mask_gau_stack[iter-1]+(1-mask_gau_stack[iter-1])*B_gau_stack_b[iter-1]).astype(np.float64)
first_g=(A_gau_stack_g[iter-1]*mask_gau_stack[iter-1]+(1-mask_gau_stack[iter-1])*B_gau_stack_g[iter-1]).astype(np.float64)
first_r=(A_gau_stack_r[iter-1]*mask_gau_stack[iter-1]+(1-mask_gau_stack[iter-1])*B_gau_stack_r[iter-1]).astype(np.float64)

for i in range(iter):
  first_b+=out_put_lap_b[i]
  first_g+=out_put_lap_g[i]
  first_r+=out_put_lap_r[i]

first_b=norm(first_b)
first_g=norm(first_g)
first_r=norm(first_r)
first=cv2.merge((first_b,first_g,first_r))
cv2.imwrite('res10.jpg',first)



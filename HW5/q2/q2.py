import numpy as np
import cv2

from scipy import sparse
from scipy.sparse.linalg import spsolve
src_center=(390,456)

img_source = cv2.imread('res05.jpg').astype(np.int16)

img_target = cv2.imread('res06.jpg')
mask = cv2.imread('mask.jpg')
row1, col1, h1 = img_target.shape
row2, col2, h2= img_source.shape
p=mask[:,:,0].copy()
c=p.copy()
p[p<=255//2]=0
p[p>255//2]=1
mask=p.astype(np.uint8)


lap_kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
lap_source_b=cv2.filter2D(img_source[:,:,0],-1,lap_kernel)
lap_source_g=cv2.filter2D(img_source[:,:,1],-1,lap_kernel)
lap_source_r=cv2.filter2D(img_source[:,:,2],-1,lap_kernel)
lap_img=cv2.merge((lap_source_b,lap_source_g,lap_source_r))

kernel_erode = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
smaller_image = cv2.erode(mask, kernel_erode, iterations=1)
boundaries = mask - smaller_image
mat_size = np.size(np.where(mask==1)[0])

unknown_mat = np.zeros([mat_size, mat_size], dtype=np.int8)


[rows, cols] = mask.shape
linear_lap = np.zeros([mat_size, 3])
index = 1
mapper = np.zeros((row2,col2)).astype(np.uint32)
inverse_mapper = np.zeros([2, mat_size]).astype(np.uint32)

for i in range(rows):
    for j in range(cols):
        if (mask[i, j] == 1):
            mapper[i, j] = index
            linear_lap[index - 1,:] = lap_img[i, j, :]
            inverse_mapper[0, index - 1] = i
            inverse_mapper[1, index - 1] = j
            index = index + 1


[rows, cols] = boundaries.shape
for i in range(rows):
    for j in range(cols):
        if (boundaries[i, j] == 1):
          try:
            p=src_center[0] - row2 // 2
            t=src_center[1] - col2 // 2
            unknown_mat[mapper[i, j] - 1, mapper[i, j] - 1] = -4
            if (mask[i + 1, j] == 1):
                unknown_mat[mapper[i, j] - 1, mapper[i + 1, j] - 1] = 1
            else:
                linear_lap[mapper[i, j] - 1, :] -= img_target[p+i + 1, t+j, :]
            if( mask[i - 1, j] == 1):
                unknown_mat[mapper[i, j] - 1, mapper[i - 1, j] - 1] = 1
            else:
                linear_lap[mapper[i, j] - 1, :] -=  img_target[p+i - 1, t+j, :]
            if (mask[i, j + 1] == 1):
                unknown_mat[mapper[i, j] - 1, mapper[i, j] + 1 - 1] = 1
            else:
                linear_lap[mapper[i, j] - 1, :] -=  img_target[p+i, j + t+1, :]
            if( mask[i, j - 1] == 1):
                unknown_mat[mapper[i, j] - 1, mapper[i, j] - 2] = 1
            else:
                linear_lap[mapper[i, j] - 1, :] -=  img_target[p+i, t+j - 1, :]
          except:
            print('1')
        elif(smaller_image[i, j] == 1):
            unknown_mat[mapper[i, j] - 1, mapper[i, j] - 1] = -4
            unknown_mat[mapper[i, j] - 1, mapper[i + 1, j] - 1] = 1
            unknown_mat[mapper[i, j] - 1, mapper[i - 1, j] - 1] = 1
            unknown_mat[mapper[i, j] - 1, mapper[i, j] - 1 + 1] = 1
            unknown_mat[mapper[i, j] - 1, mapper[i, j] - 1 - 1] = 1



coef_mat = sparse.csc_matrix(unknown_mat)
lap_b = sparse.csc_matrix(np.reshape(linear_lap[:, 0], (mat_size,1 )))
lap_g = sparse.csc_matrix(np.reshape(linear_lap[:, 1], (mat_size, 1)))
lap_r = sparse.csc_matrix(np.reshape(linear_lap[:, 2], (mat_size, 1)))


b_channel = spsolve(coef_mat, lap_b)
g_channel = spsolve(coef_mat, lap_g)
r_channel = spsolve(coef_mat, lap_r)

img=np.zeros_like(img_source)
fin_b = img[:, :, 0]
fin_g = img[:, :, 1]
fin_r = img[:, :, 2]

for i in range(mat_size):
    fin_b[inverse_mapper[0,i],inverse_mapper[1,i]]=b_channel[i]
    fin_g[inverse_mapper[0,i], inverse_mapper[1,i]] = g_channel[i]
    fin_r[inverse_mapper[0,i],inverse_mapper[1,i]]=r_channel[i]

fin_b[fin_b<=0]=0
fin_g[fin_g<=0]=0
fin_r[fin_r<=0]=0

final=cv2.merge((fin_b,fin_g,fin_r))


fin_mask=cv2.merge((smaller_image,smaller_image,smaller_image))

row2, col2, h1 = np.shape(final)
row1, col1, h2 = np.shape(img_target)
min_row=src_center[0] - row2 // 2
max_row=src_center[0] - row2 // 2 + row2
min_col=src_center[1] - col2 // 2
max_col=src_center[1]- col2 // 2 + col2
image = np.copy(img_target)
image[min_row:max_row,min_col:max_col] = fin_mask * final + (1 - fin_mask) * img_target[min_row:max_row,min_col:max_col]
cv2.imwrite('res07.jpg', image)
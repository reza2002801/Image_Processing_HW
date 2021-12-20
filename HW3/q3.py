import cv2
import numpy as np
import random

def shoetestpath_H(im1,im2):

    img=((im1 - im2) * (im1 - im2)).astype(np.float64)
    row,col,h=im1.shape


    shortPath=np.zeros((row,col))
    # shortPath.fill(np.inf)
    shortPath[:,0]=img[:,0,0]+img[:,0,1]+img[:,0,2]


    for i in range(1,col):
        for j in range(0,row):
            if(j==0):
                val=img[0,i,0]+img[0,i,1]+img[0,i,2]
                val0=val+shortPath[0,i-1]
                valmin1 = val + shortPath[1, i - 1]
                if(val0>=valmin1):
                    shortPath[0,i]=valmin1

                else:
                    shortPath[0, i] = val0


            elif(j==row-1):
                val = img[row-1, i, 0] + img[row-1, i, 1] + img[row-1, i, 2]
                val0 = val + shortPath[row-1, i - 1]
                valplus1 = val + shortPath[row-2, i - 1]
                if (val0 >= valplus1):
                    shortPath[row-1, i] = valplus1

                else:
                    shortPath[row-1, i] = val0


            else:
                val = img[j, i, 0] + img[j, i, 1] + img[j, i, 2]
                val0 = val + shortPath[j, i - 1]
                valmin1 = val + shortPath[j-1, i - 1]
                valplus1 = val + shortPath[j+1, i - 1]
                l=[val0,valmin1,valplus1]
                min=np.inf
                index=-1
                for k in range(3):
                    if(l[k]<min):
                        min=l[k]
                        index=k
                if(index==0):
                    shortPath[j, i] = val0

                elif(index==1):
                    shortPath[j, i] = valmin1

                elif (index == 2):
                    shortPath[j, i] = valplus1

                else:
                    print('Oh No')
    temp=np.inf
    index=-1
    for i in range(row-1):
        # print(shortPath[i,col-1])
        if(shortPath[i,col-1]<=temp):
            # print('d')
            index=i
            temp=shortPath[i,col-1]
    finalpath_mask =np.zeros((row,col))
    direction=-2
    # print(index)
    # print(path)

    for i in range(col-1,-1,-1):
        finalpath_mask[:index, i] = 1
        if (index == 0):

            res = findsmallest(shortPath[index, i - 1], shortPath[index + 1, i - 1], np.Inf)
            if (res == 1):
                index = index
            elif (res == 2):
                index = index + 1
        elif (index == row - 1):
            res = findsmallest(shortPath[index, i - 1], shortPath[index - 1, i - 1], np.Inf)
            if (res == 1):
                index = index
            elif (res == 2):
                index = index - 1
        else:
            res = findsmallest(shortPath[index, i - 1], shortPath[index - 1, i - 1], shortPath[index + 1, i - 1])
            if (res == 1):
                index = index
            elif (res == 2):
                index = index - 1
            elif (res == 3):
                index = index + 1
    n=np.zeros((row,col,3))
    n[:,:,0]=finalpath_mask
    n[:, :, 1] = finalpath_mask
    n[:, :, 2] = finalpath_mask

    return n
def findsmallest(a,b,c):
    if(a<=b and a<=c):
        return 1
    if(b<=a and b<=c):
        return 2
    if(c<=a and c<=b):
        return 3
def random_L_template_matching(img_vertical,img_horizontal,img_square,texture_image,rectangle_around,patch_size):
    row,col,j=texture_image.shape
    t_match1 = cv2.matchTemplate(texture_image.astype(np.uint8), img_vertical.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
    t_match2 = cv2.matchTemplate(texture_image.astype(np.uint8), img_horizontal.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
    t_match3 = cv2.matchTemplate(texture_image.astype(np.uint8), img_square.astype(np.uint8), cv2.TM_CCOEFF_NORMED)

    m1 = np.zeros((row, col), np.float64)
    m2 = np.zeros((row, col), np.float64)
    m3 = np.zeros((row, col), np.float64)



    m1[0: t_match1.shape[0], 0: t_match1.shape[1]] = t_match1
    m2[0: t_match2.shape[0], 0: t_match2.shape[1]] = t_match2
    m3[0: t_match3.shape[0], 0: t_match3.shape[1]] = t_match3

    mask = m3.copy()
    mask[0: row, 0: col - common_size] += m1[0: row, common_size: col]
    mask[0: row - common_size, 0: col] += m2[common_size: row, 0: col]

    zs=mask-np.min(mask)
    zs/=np.max(zs)
    zs[row-patch_size:,:]=0
    zs[:,col-patch_size:]=0
    listofmatchPoints = []

    for i in range(100):
        p = list(zip(np.where(zs == np.max(zs))[1], np.where(zs == np.max(zs))[0]))
        listofmatchPoints.append(p)
        zs[p[0][1], p[0][0]] = 0

    h=0
    for i in range(100):

        x, y = (listofmatchPoints[i][0][1], listofmatchPoints[i][0][0])
        if not(x>rectangle_around[0][0]-patch_size and x<rectangle_around[1][0] and y>rectangle_around[0][1]-patch_size and y<rectangle_around[1][1] ):
            h=i
            break
        if(i>=99):
            x, y = (listofmatchPoints[80][0][1], listofmatchPoints[80][0][0])
    # rnd=random.randint(h,80

    return x,y
def shoetestpath_V(im1,im2):
    img = ((im1 - im2) * (im1 - im2)).astype(np.float64)
    row, col, h = im1.shape
    path = np.zeros((row, col))
    path.fill(np.inf)
    shortPath = np.zeros((row, col),np.float64)
    shortPath[0, :] = img[0, :, 0] + img[0, :, 1] + img[0, :, 2]
    path[0,:]=0
    for i in range(1,row):
        for j in range(0,col):
            if (j == 0):
                val = img[i, 0, 0] + img[i, 0, 1] + img[i, 0, 2]
                val0 = val + shortPath[i-1, 0]
                valplus1 = val + shortPath[i-1,1]
                if (val0 >= valplus1):
                    shortPath[i, 0] = valplus1
                    path[i, 0] = 1
                else:
                    shortPath[i,0] = val0
                    path[i,0] = 0

            elif (j == col - 1):
                val = img[ i,col - 1, 0] + img[i,col - 1, 1] + img[i,col - 1, 2]
                val0 = val + shortPath[i-1, col-1]
                valmin1 = val + shortPath[i-1,col-2]
                if (val0 >= valmin1):
                    shortPath[i, col-1] = valmin1
                    path[i, col-1] = -1
                else:
                    shortPath[i, col-1] = val0
                    path[i, col-1]= 0

            else:
                val = img[i,j, 0] + img[i,j, 1] + img[i,j, 2]
                val0 = val + shortPath[i-1,j]
                valmin1 = val + shortPath[i-1, j - 1]
                valplus1 = val + shortPath[i-1, j + 1]
                l = [val0, valmin1, valplus1]
                min = np.inf
                index = -1
                for k in range(3):
                    if (l[k] < min):
                        min = l[k]
                        index = k
                if (index == 0):
                    shortPath[i, j] = val0
                    path[i, j] = 0
                elif (index == 1):
                    shortPath[i, j] = valmin1
                    path[i, j] = -1
                elif (index == 2):
                    shortPath[i, j] = valplus1
                    path[i, j] = 1
                else:
                    print('Oh No')

    temp = np.inf
    index = -1
    for i in range(col - 1):

        if (shortPath[row-1, i] <= temp):

            index = i
            temp = shortPath[row-1, i]
    finalpath_mask = np.zeros((row, col))

    for i in range(row-1,-1,-1):
        finalpath_mask[i, :index] = 1
        if (index == 0):

            res = findsmallest(shortPath[i-1, index], shortPath[i - 1, index+1], np.Inf)
            if (res == 1):
                index = index
            elif (res == 2):
                index = index + 1
        elif (index == col - 1):
            res = findsmallest(shortPath[i-1, index], shortPath[i-1,index-1], np.Inf)
            if (res == 1):
                index = index
            elif (res == 2):
                index = index - 1
        else:
            res = findsmallest(shortPath[i-1, index], shortPath[i-1, index-1], shortPath[i-1, index+1])
            if (res == 1):
                index = index
            elif (res == 2):
                index = index - 1
            elif (res == 3):
                index = index + 1

    n=np.zeros((row,col,3))
    n[:,:,0]=finalpath_mask
    n[:, :, 1] = finalpath_mask
    n[:, :, 2] = finalpath_mask

    return n

def fill_Hole(rectangle_around,img,patch_size):
    patch_size=int(patch_size*int(np.round(img.shape[0]/960)))
    texture=img.copy()
    final_img=img.copy()
    horiz_n=0
    vertic_n=0
    for i in range(rectangle_around[0][0]-common_size, rectangle_around[1][0] -common_size , patch_size - common_size):
        for j in range(rectangle_around[0][1]-common_size, rectangle_around[1][1] -common_size , patch_size - common_size):
            if(i==rectangle_around[0][0]-common_size):
                horiz_n+=1
            if(j==rectangle_around[0][1]-common_size):
                vertic_n+=1
            H_part_L = final_img[i: i + common_size, j + common_size: j + patch_size, :].copy()
            s_part_L = final_img[i: i + common_size, j: j + common_size, :].copy()
            v_part_L = final_img[i + common_size: i + patch_size, j: j + common_size, :].copy()



            original = final_img[i: i + patch_size, j: j + patch_size, :].copy()
            x, y = random_L_template_matching(H_part_L, v_part_L, s_part_L,texture,rectangle_around,patch_size)
            new_v_L = texture[x + common_size: x + patch_size, y: y + common_size,:].copy()
            new_h_L = texture[x: x + common_size, y + common_size: y + patch_size,:].copy()
            square_replace_patch = texture[x: x + patch_size, y: y + patch_size, :].copy()

            v_m = shoetestpath_V(v_part_L, new_v_L)
            h_m = shoetestpath_H(H_part_L, new_h_L)
            v_mask = np.zeros((patch_size, patch_size, 3), np.uint8)
            v_mask[0: v_m.shape[0], 0: v_m.shape[1], :] = v_m
            h_mask = np.zeros((patch_size, patch_size, 3),  np.uint8)
            h_mask[0: h_m.shape[0], 0: h_m.shape[1], :] = h_m
            mask=v_mask+h_mask
            ones = np.ones((patch_size,patch_size,3),  np.uint8)
            mask2=(ones - mask)
            final_img[i: i + patch_size, j: j + patch_size, :] = mask * original +  mask2* square_replace_patch

    final_img1=final_img.astype(np.uint8)
    print(final_img1.shape)
    t2=texture.astype(np.uint8)
    w=rectangle_around[1][1]-rectangle_around[0][1]
    h=rectangle_around[1][0]-rectangle_around[0][0]
    extra_h=horiz_n*(patch_size-common_size)-(rectangle_around[1][1]-rectangle_around[0][1])
    extra_v=vertic_n*(patch_size-common_size)-(rectangle_around[1][0]-rectangle_around[0][0])
    v_part_extra=t2[rectangle_around[1][0]-h:rectangle_around[1][0],rectangle_around[1][1]:rectangle_around[1][1]+extra_h,:].copy()
    new=final_img1[rectangle_around[1][0]-h:rectangle_around[1][0],rectangle_around[1][1]:rectangle_around[1][1]+extra_h,:].copy()


    verticalMask = shoetestpath_V(v_part_extra, new)
    
    ones = np.ones(new.shape, dtype='uint8')
    temp_res = verticalMask * new + (ones - verticalMask) * v_part_extra

    h_part_extra=t2[rectangle_around[1][0]:rectangle_around[1][0]+extra_v,rectangle_around[0][1]-common_size:rectangle_around[1][1]+extra_h,:].copy()
    new=final_img1[rectangle_around[1][0]:rectangle_around[1][0]+extra_v,rectangle_around[0][1]-common_size:rectangle_around[1][1]+extra_h,:].copy()
    H_mask=shoetestpath_H(new,h_part_extra)
    ones = np.ones(new.shape, dtype='uint8')
    H_mask2= (ones - H_mask)
    final_img[rectangle_around[1][0]:rectangle_around[1][0]+extra_v,rectangle_around[0][1]-common_size:rectangle_around[1][1]+extra_h,:] = H_mask * new +H_mask2* h_part_extra
   
    final_img[rectangle_around[1][0]-h:rectangle_around[1][0],rectangle_around[1][1]:rectangle_around[1][1]+extra_h,:]=temp_res

    return final_img

img=cv2.imread('im03.jpg')
img2=cv2.imread('im04.jpg')


texture=img.copy()
final_img=img.copy()



rec_around_birds=[[(61,319),(170,550)],[(605,1128),(720,1254)],[(744,805),(930,975)]]
rec_around_human=[[(699,742),(1164,950)]]

patch_size=45
common_size=15

def final(list,img,patch_size):
    f=img.copy()
    for i in list:
        f=fill_Hole(i,f,patch_size)
    return f


cv2.imwrite('res15.jpg',final(rec_around_birds,img,patch_size))
cv2.imwrite('res16.jpg',final(rec_around_human,img2,patch_size))


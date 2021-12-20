
import cv2
import numpy as np
import random



def make_first_patch(texture):
    row,col,g=texture.shape
    x = random.randint(0, col - patch_size)
    y = random.randint(0, row - patch_size)
    first_part = texture[y:y + patch_size, x:x + patch_size, :].copy()
    return first_part



def random_L_template_matching(img_vertical,img_horizontal,img_square,texture_image):
    row, col, j = texture_image.shape
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

    zs = mask - np.min(mask)
    zs /= np.max(zs)
    zs[row - patch_size:, :] = 0
    zs[:, col - patch_size:] = 0
    listofmatchPoints = []
    for i in range(10):
        p = list(zip(np.where(zs == np.max(zs))[1], np.where(zs == np.max(zs))[0]))
        listofmatchPoints.append(p)
        zs[p[0][1], p[0][0]] = 0
    rnd = random.randint(0, 9)

    return listofmatchPoints[rnd][0][1], listofmatchPoints[rnd][0][0]

def radom_template_match(img,texture):
    row,col,h=texture.shape
    template_match_res=cv2.matchTemplate(texture.astype(np.uint8),img.astype(np.uint8),cv2.TM_CCOEFF_NORMED)
    row1,col1=template_match_res.shape

    f=np.zeros((row,col))
    f[:row1,:col1]=template_match_res
    f[:,col-patch_size:]=0
    f[row-patch_size:,:]=0
    listofmatchPoints=[]
    for i in range(10):
        p=list(zip(np.where(f==np.max(f))[1],np.where(f==np.max(f))[0]))
        listofmatchPoints.append(p)
        f[p[0][1],p[0][0]]=0
    rnd=random.randint(0,9)
    print(len(listofmatchPoints))



    return listofmatchPoints[rnd][0][1],listofmatchPoints[rnd][0][0]

def shoetestpath_H(image1,image2):

    img=((image1 - image2) * (image1 - image2))
    row,col,h=image1.shape


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

def shoetestpath_V(image1,image2):
    img = (image1 - image2) * (image1 - image2)
    row, col, h = image1.shape
    path = np.zeros((row, col))
    path.fill(np.inf)
    shortPath = np.zeros((row, col))
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
        # print(shortPath[i,col-1])
        if (shortPath[row-1, i] <= temp):
            # print('d')
            index = i
            temp = shortPath[row-1, i]
    finalpath_mask = np.zeros((row, col))
    print(index)
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




def texture_synthes(texture):
    row,col,g=texture.shape
    final_img = np.zeros((2600, 2600, 3))
    final_img[:   patch_size, :  patch_size, :] = make_first_patch(texture)
    for j in range(0, 2600 - patch_size + 1, patch_size - common_size):
        temp = np.zeros((patch_size, patch_size, 3), dtype='uint8')
        if  j!=0:
            V_Part = final_img[0: 0 + patch_size, j: j + common_size, :].copy()

            x, y = radom_template_match(V_Part.astype(np.uint8),texture)
            print('fd')
            common_part = texture[x: x + patch_size, y: y + common_size, :].copy()
            v_m = shoetestpath_V(V_Part, common_part)
            print(v_m.shape,'dsws')
            ones = np.ones((patch_size, common_size, 3),  np.uint8)
            fin_img = (v_m * V_Part) + ((ones - v_m) * common_part)
            temp = texture[x: x + patch_size, y:y + patch_size, :].copy()
            temp[0: patch_size, 0: common_size, :] = fin_img
            final_img[0: 0 + patch_size, j: j + patch_size, :] = temp

    for i in range(0, 2600 - patch_size + 1, patch_size - common_size):
        temp = np.zeros((patch_size, patch_size, 3), np.uint8)
        if  i!=0:
            H_part = final_img[i: i + common_size, 0: 0 + patch_size, :].copy()
            x, y = radom_template_match(H_part.astype(np.uint8),texture)
            H_common = texture[x: x + common_size, y: y + patch_size, :].copy()
            h_m = shoetestpath_H(H_part, H_common)
            fin_img = np.ones((common_size, patch_size, 3),  np.uint8)
            fin_img = h_m * H_common + (
                    fin_img - h_m) * H_part
            temp = texture[x: x + patch_size, y: y + patch_size, :].copy()
            temp[0: common_size, 0: patch_size, :] = fin_img
            final_img[i: i + patch_size, 0: 0 + patch_size, :] = temp

    for i in range(0, 2600 - patch_size + 1, patch_size - common_size):
        for j in range(0, 2600 - patch_size + 1, patch_size - common_size):
            if j != 0 and i!=0:

                H_part_L = final_img[i: i + common_size, j + common_size: j + patch_size, :].copy()
                s_part_L = final_img[i: i + common_size, j: j + common_size, :].copy()
                v_part_L = final_img[i + common_size: i + patch_size, j: j + common_size, :].copy()

                original = final_img[i: i + patch_size, j: j + patch_size, :].copy()
                x, y = random_L_template_matching(H_part_L, v_part_L, s_part_L, texture)
                new_v_L = texture[x + common_size: x + patch_size, y: y + common_size, :].copy()
                new_h_L = texture[x: x + common_size, y + common_size: y + patch_size, :].copy()
                square_replace_patch = texture[x: x + patch_size, y: y + patch_size, :].copy()

                v_m = shoetestpath_V(v_part_L, new_v_L)
                h_m = shoetestpath_H(H_part_L, new_h_L)
                v_mask = np.zeros((patch_size, patch_size, 3), np.uint8)
                v_mask[0: v_m.shape[0], 0: v_m.shape[1], :] = v_m
                h_mask = np.zeros((patch_size, patch_size, 3), np.uint8)
                h_mask[0: h_m.shape[0], 0: h_m.shape[1], :] = h_m
                mask = np.bitwise_or(v_mask, h_mask)
                ones = np.ones((patch_size, patch_size, 3), np.uint8)
                mask2 = (ones - mask)
                final_img[i: i + patch_size, j: j + patch_size, :] = mask * original + mask2 * square_replace_patch



    h=np.zeros((2500,2500+col,3))
    h[:2500,:2500,:]=final_img[:2500,:2500,:]
    h[:row,2500:,:]=texture
    return h
texture1=cv2.imread('textureMy01.jpg')
texture2=cv2.imread('textureMy02.jpg')
texture3=cv2.imread('texture06.jpg')
texture4=cv2.imread('texture11.jpeg')


patch_size=130# size of the patch which is patch_size*patch_size
common_size=30# size of the region which intersects with the built part of the image


cv2.imwrite('res11.jpg',texture_synthes(texture1))
cv2.imwrite('res12.jpg',texture_synthes(texture2))
cv2.imwrite('res13.jpg',texture_synthes(texture3))
cv2.imwrite('res14.jpg',texture_synthes(texture4))





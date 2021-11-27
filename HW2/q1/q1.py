

import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('flowers.blur.png')
def la(ker_size,sigma):
    res=np.zeros((ker_size,ker_size))
    mean =(ker_size//2)
    for i in range(ker_size):
        for j in range(ker_size):
            res[i,j]=(((i-mean)**2+(j-mean)**2-2*sigma**2)/(2*3.14*np.power(sigma,6)))*\
                     np.exp(-((i-mean)**2+(j-mean)**2)/(2*sigma**2))-(1/(3.14*np.power(sigma,4)))
    # res = res / np.sum(res)
    return res

def gausian_filter_maker(ker_size,sigma):
    res=np.zeros((ker_size,ker_size))
    mean =(ker_size//2)
    for i in range(ker_size):
        for j in range(ker_size):
            res[i,j]=np.exp(-((i-mean)**2+(j-mean)**2)/(2*sigma**2))
    res = res / np.sum(res)
    return res

def gausian_filter_fourier(D0,dim):
    res = np.ones(dim)
    p=np.zeros(dim)
    mean_x=dim[0]//2
    mean_y = dim[1] // 2
    for i in range(dim[0]):
        for j in range(dim[1]):
            p[i,j]=np.exp(-((i-mean_x)**2+(j-mean_y)**2)/(2*D0**2))
    res=res-p
    # cv2.imwrite('j.jpg',255*res)
    return res


def unsharp_mask(img):
    alpha = 2
    kernel = gausian_filter_maker(9,3)
    # k2=255*kernel
    k=kernel-np.min(kernel)
    k/=np.max(k)
    k=(255*k).astype(np.uint8)
    c= cv2.merge((k,k,k))
    row,col,h=c.shape
    c = cv2.resize(c, (40*row,40*col), interpolation=cv2.INTER_AREA)

    cv2.imwrite('res01.jpg',c)

    img_smooth = cv2.filter2D(img, -1, kernel).astype(np.float64)
    cv2.imwrite('res02.jpg', img_smooth)
    img_minus = (img - img_smooth).astype(np.float64)
    print(np.mean(img_minus))
    cv2.imwrite('res03.jpg', img_minus)
    img_final = (img + alpha * img_minus)
    cv2.imwrite('res04.jpg',img_final)

def laplacian(img):
    k=4
    kernel = la(15,1)

    show=kernel-np.min(kernel)
    show=show/np.max(show)

    n=kernel-np.mean(kernel)
    # normalized_v = n / np.sqrt(np.sum(n ** 2))
    print(255*show)
    row, col = show.shape
    show = cv2.resize(show, (40 * row, 40 * col), interpolation=cv2.INTER_AREA)
    cv2.imwrite('res05.jpg', 255*show)
    lap_img=cv2.filter2D(img,-1,n).astype(np.float64)

    lap_img2=lap_img-np.min(lap_img)
    lap_img2/=np.max(lap_img2)
    # print(lap_img)
    cv2.imwrite('res06.jpg', 255*lap_img2)
    img_final=(img-k*lap_img).astype(np.float64)
    cv2.imwrite('res07.jpg', img_final)

def fourier_1(img):
    fimg_b = go_to_freq_domain(img[:,:,0])
    fimg_g = go_to_freq_domain(img[:, :, 1])
    fimg_r = go_to_freq_domain(img[:, :, 2])

    mask=gausian_filter_fourier(75,(img.shape[0],img.shape[1]))

    k=0.6
    mask_img_b = k*(fimg_b * mask)+fimg_b
    mask_img_g = k*(fimg_g * mask)+fimg_g
    mask_img_r = k*(fimg_r * mask)+fimg_r



    simg_b=go_to_special_domain(mask_img_b)
    simg_g = go_to_special_domain(mask_img_g)
    simg_r = go_to_special_domain(mask_img_r)


    mfb1 = np.abs(mask_img_b)
    mfg1 = np.abs(mask_img_g)
    mfr1 = np.abs(mask_img_r)
    mfb1 = np.log(mfb1)
    mfg1 = np.log(mfg1)
    mfr1 = np.log(mfr1)
    mfb1 -= np.min(mfb1)
    mfg1 -= np.min(mfg1)
    mfr1 -= np.min(mfr1)

    mfb1 /= np.max(mfb1)
    mfg1 /= np.max(mfg1)
    mfr1 /= np.max(mfr1)
    d = cv2.merge((mfb1,mfg1,mfr1))

    new_image=cv2.merge((simg_b,simg_g,simg_r))

    fb1=np.abs(fimg_b)
    fg1 = np.abs(fimg_g)
    fr1 = np.abs(fimg_r)
    fb1=np.log(fb1)
    fg1=np.log(fg1)
    fr1=np.log(fr1)
    fb1-=np.min(fb1)
    fg1-=np.min(fg1)
    fr1-=np.min(fr1)

    fb1/=np.max(fb1)
    fg1/=np.max(fg1)
    fr1/=np.max(fr1)
    c=cv2.merge((fb1,fg1,fr1))
    cv2.imwrite('res08.jpg',255*c)
    cv2.imwrite('res09.jpg',255*mask)
    cv2.imwrite('res10.jpg',255*d)
    cv2.imwrite('res11.jpg',new_image)
def fourier_2(img):
    fimg_b = go_to_freq_domain(img[:, :, 0])
    fimg_g = go_to_freq_domain(img[:, :, 1])
    fimg_r = go_to_freq_domain(img[:, :, 2])

    filter = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)


    p=img.shape[0] / 2
    p2=img.shape[1] / 2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filter[i][j] = (i-p)**2+(j-p2)**2

    k=0.0000008

    filter=4*3.14*3.14*(filter)

    mask_img_b=filter*fimg_b
    mask_img_g = filter * fimg_g
    mask_img_r = filter * fimg_r

    simg_b=go_to_special_domain(mask_img_b)
    simg_g = go_to_special_domain(mask_img_g)
    simg_r = go_to_special_domain(mask_img_r)

    new_image=cv2.merge((simg_b,simg_g,simg_r))

    im=img.astype(np.float64)
    im=im+k*new_image

    c=cv2.merge((np.abs(mask_img_b),np.abs(mask_img_g),np.abs(mask_img_r)))
    cv2.imwrite('res12.jpg',3*np.log(c))

    new_image-=np.min(new_image)
    new_image/=np.max(new_image)



    cv2.imwrite('res13.jpg',255*new_image)
    cv2.imwrite('res14.jpg',im)
def go_to_freq_domain(img):
    im_fft = np.fft.fft2(img)
    shifted_image = np.fft.fftshift(im_fft)

    return shifted_image

def go_to_special_domain(img):
    fil_im_ishifted = np.fft.ifftshift(img)
    fil_im = np.fft.ifft2(fil_im_ishifted)
    fil_im = np.real(fil_im)

    return fil_im

unsharp_mask(img)
laplacian(img)
fourier_2(img)
fourier_1(img)

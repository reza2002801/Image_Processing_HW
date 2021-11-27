import cv2
import numpy as np
import math
img_near=cv2.imread('res19-near.jpg')
img_far=cv2.imread('res20-far.jpg')
row,col,h= img_near.shape
dim=(col,row)
img_far = cv2.resize(img_far, dim, interpolation=cv2.INTER_AREA)


near = np.array([
    (647, 929),
    (1105, 921),
    (873, 1173),

])
# far
far = np.array([
    (691, 777),
    (1080, 779),
    (884, 1001),

])
t_mat, _ = cv2.estimateAffine2D(near, far)

img_near = cv2.warpAffine(img_near.astype(np.float64), t_mat, img_far.shape[:2][::-1], flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
cv2.imwrite('res21-near.jpg',img_near)
cv2.imwrite('res22-far.jpg',img_far)



def gauss_filter(dims, sigma):
    res = np.zeros(dims)
    mio = np.array(dims) / 2
    for loc in np.ndindex(dims):
        temp = np.array(loc) - np.array(mio)
        res[loc] = math.exp(-((temp * temp).sum()) / sigma ** 2 / 2) / (2 * math.pi * sigma ** 2)
    return res / res.max()



def show_func(mask_img_b,mask_img_g,mask_img_r):
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
    return 255*cv2.merge((mfb1,mfg1,mfr1))
def show_func2(mask_img_b,mask_img_g,mask_img_r):
    mfb1 = np.abs(mask_img_b)
    mfg1 = np.abs(mask_img_g)
    mfr1 = np.abs(mask_img_r)
    mfb1 = np.log(mfb1)
    mfg1 = np.log(mfg1)
    mfr1 = np.log(mfr1)

    mfb1 /= np.max(mfb1)
    mfg1 /= np.max(mfg1)
    mfr1 /= np.max(mfr1)
    return 255*cv2.merge((mfb1,mfg1,mfr1))
def go_to_freq_domain(img):
    im_fft = np.fft.fft2(img)
    shifted_image = np.fft.fftshift(im_fft)

    return shifted_image
def go_to_special_domain(img):
    fil_im_ishifted = np.fft.ifftshift(img)
    fil_im = np.fft.ifft2(fil_im_ishifted)
    fil_im = np.real(fil_im)

    return fil_im

lowsigma = 36
highsigma = 50

lowpass_filter = gauss_filter(img_far.shape[:2], lowsigma)
highpass_filter = 1 - gauss_filter(img_near.shape[:2], highsigma)


fimg_b=go_to_freq_domain(img_far[:,:,0])
fimg_g = go_to_freq_domain(img_far[:, :, 1])
fimg_r = go_to_freq_domain(img_far[:, :, 2])


nimg_b=go_to_freq_domain(img_near[:,:,0])
nimg_g = go_to_freq_domain(img_near[:, :, 1])
nimg_r = go_to_freq_domain(img_near[:, :, 2])
nn=cv2.merge((np.abs(nimg_b),np.abs(nimg_g),np.abs(nimg_r)))
mm=cv2.merge((np.abs(fimg_b),np.abs(fimg_g),np.abs(fimg_r)))
cv2.imwrite('res23-dft-near.jpg',show_func(nimg_b,nimg_g,nimg_r))
cv2.imwrite('res24-dft-far.jpg',show_func(fimg_b,fimg_g,fimg_r))
cv2.imwrite('res25-highpass-50.jpg',255*highpass_filter)
cv2.imwrite('res26-lowpass-36.jpg',255*lowpass_filter)


img_nb=fimg_b*lowpass_filter
img_ng=fimg_g*lowpass_filter
img_nr=fimg_r*lowpass_filter

img_nb1=nimg_b*highpass_filter
img_ng2=nimg_g*highpass_filter
img_nr3=nimg_r*highpass_filter



cv2.imwrite('res27-highpassed.jpg',show_func2(img_nb1,img_ng2,img_nr3))
cv2.imwrite('res28-lowpassed.jpg',show_func2(img_nb,img_ng,img_nr))

img_nb=img_nb+img_nb1
img_ng=img_ng+img_ng2
img_nr=img_nr+img_nr3
cv2.imwrite('res29-hybrid.jpg',show_func(img_nb,img_ng,img_nr))

simg_b=go_to_special_domain(img_nb)
simg_g = go_to_special_domain(img_ng)
simg_r = go_to_special_domain(img_nr)

n=cv2.merge((simg_b,simg_g,simg_r))
cv2.imwrite('res30-hybrid-near.jpg',n)
n2= cv2.resize(n, (int(0.05*n.shape[1]),int(0.05*n.shape[0])), interpolation=cv2.INTER_AREA)

cv2.imwrite('res31-hybrid-far.jpg',n2)

'''
#wiener filter with average K
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import skimage

# Defining Values

blur = 5
variance = 1000

def readimage(string):
    image = cv2.imread(string,0)
    imagefinal = cv2.resize(image, (512,512))
    return imagefinal

def showimage(image):
    plot = plt.imshow(image,cmap='gray')
    plt.show(plot)
    return plot

def add_blur(image):
    imgblur = cv2.blur(image,(blur,blur))
    return imgblur

def add_noise(image):
    mean=0
    sigma = variance**0.5
    row,col= image.shape
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy_image = np.uint8(image + gauss)
    return noisy_image

def psnr(img1,img2):
    return skimage.measure.compare_psnr(img1,img2)


def psd(image):
    fourierimage = np.fft.fft2(image)
    psdimage = fourierimage * np.conj(fourierimage) / (512 * 512)
    return psdimage


def wiener_filter():
    h = np.ones((blur, blur)) / (blur * blur)
    H = np.fft.fft2(h, [512, 512])
    psdmean = 0
    for i in glob.glob("*.png"):
        grayimage = cv2.imread(i, 0)
        image = cv2.resize(grayimage, (512, 512))
        img_blur = add_blur(image)
        noisy = add_noise(img_blur)

        PSD = psd(image)
        psdmean += PSD
    psdmean = psdmean / 14
    N = variance * np.ones((512, 512))
    K = N / psdmean
    print(K)
    Filter = np.conj(H) / (H * np.conj(H) + K)

    return Filter


Filter = wiener_filter()

image=readimage('lena.png')
plt.imshow(image)
plt.show()

img_blur=add_blur(image)
noise = add_noise(img_blur)
plt.imshow(noise,cmap='gray')
plt.show()

def filter_test(noisy_image,wiener_filter):
    f=wiener_filter*np.fft.fft2(noisy_image)
    o=np.fft.ifft2(f)
    output=np.uint8(o)
    return output
result = filter_test(noise,Filter)
plt.imshow(result,cmap='gray')
plt.show()


'''
import cv2
import numpy as np

img = cv2.imread('image/lena.png')
print(img.shape)
a = [[1,2],[8,4],[5,6]]
print(np.sum(a))

def gauss_blur(img,sigma):
    '''suitable for 1 or 3 channel image'''
    row_filter=get_gauss_kernel(sigma,1)
    t=cv2.filter2D(img,-1,row_filter[...,None])
    return cv2.filter2D(t,-1,row_filter.reshape(1,-1))

def get_gauss_kernel(sigma,dim=2):
    '''1D gaussian function: G(x)=1/(sqrt{2π}σ)exp{-(x-μ)²/2σ²}. Herein, μ:=0, after
       normalizing the 1D kernel, we can get 2D kernel version by
       matmul(1D_kernel',1D_kernel), having same sigma in both directions. Note that
       if you want to blur one image with a 2-D gaussian filter, you should separate
       it into two steps(i.e. separate the 2-D filter into two 1-D filter, one column
       filter, one row filter): 1) blur image with first column filter, 2) blur the
       result image of 1) with the second row filter. Analyse the time complexity: if
       m&n is the shape of image, p&q is the size of 2-D filter, bluring image with
       2-D filter takes O(mnpq), but two-step method takes O(pmn+qmn)'''
    ksize=int(np.floor(sigma*6)/2)*2+1 #kernel size("3-σ"法则) refer to
    #https://github.com/upcAutoLang/MSRCR-Restoration/blob/master/src/MSRCR.cpp
    k_1D=np.arange(ksize)-ksize//2
    k_1D=np.exp(-k_1D**2/(2*sigma**2))
    k_1D=k_1D/np.sum(k_1D)
    if dim==1:
        return k_1D
    elif dim==2:
        return k_1D[:,None].dot(k_1D.reshape(1,-1))

import skimage
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

origin = cv2.imread("image/lena.png")
origin = gauss_blur(origin,sigma=8)
noisy = skimage.util.random_noise(origin, mode='gaussian', var=0.05)
noisy = img_as_ubyte(noisy)
cv2.imwrite('image/lena_noisy.png',np.array(noisy))
plt.imshow(noisy)
plt.show()

'''
img = 'image/lena.png'
img = cv2.imread(img)
re = gauss_blur(img,sigma=8)
cv2.imwrite('t.png',re)
'''
'''
from skimage import io, img_as_float
from skimage import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
mean = 0
var = 0.01

image = io.imread("image/lena.png")

image = img_as_float(image)
noise = np.random.normal(mean, var**0.5, image.shape)
noisy = image + noise
noisy = np.clip(noisy, 0.0, 1.0)
plt.imshow(noisy)
plt.show()
noisy = img_as_ubyte(noisy)
print(noisy,noisy.shape)
img = [None]*3
for k in range(3):
    img[k] = noisy[:,:,k]
print(img)
cv2.imwrite('t.png',np.array(img))
'''
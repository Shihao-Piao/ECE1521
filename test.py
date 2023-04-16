
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
import os
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt
import cv2

def readimage(string):
    image = cv2.imread(string, 0)
    imagefinal = cv2.resize(image, (512, 512))
    return imagefinal


def blur(img, kernel_size=3):
    imgblur = cv2.blur(img, (kernel_size, kernel_size))
    return imgblur


def add_gaussian_noise(img, sigma):
    gauss = np.random.normal(0, sigma, np.shape(img))
    noisy_img = img + gauss
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img


def wiener_filter(img_path, kernel_size = 9, K = 10):
    img = readimage(img_path)
    kernel = gaussian_kernel(kernel_size)
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy


def gaussian_kernel(kernel_size=3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == '__main__':
    path = 'image/lena_noisy.png'
    img = cv2.imread(path,0)
    plt.imshow(img,cmap='gray')
    plt.show()
    re = wiener_filter(path)
    print(re.astype('uint8'))
    fig = plt.figure()
    plt.imshow(re,cmap='gray')
    plt.savefig('t.png')
    plt.show()
    #cv2.imwrite('t.png',re)

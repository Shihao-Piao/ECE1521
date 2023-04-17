import numpy as np
from matplotlib import pyplot as plt
import cv2


def readimage(string):
    image = cv2.imread(string, 0)
    imagefinal = cv2.resize(image, (512, 512))
    return imagefinal


def calc_histogram(gray_arr, level=256):
    hists = [0 for _ in range(level)]
    for row in gray_arr:
        for p in row:
            hists[p] += 1
    return hists


def calc_histogram_cdf(hists, block_m, block_n, level=256):
    hists_cumsum = np.cumsum(np.array(hists))
    const_a = (level - 1) / (block_m * block_n)
    hists_cdf = (const_a * hists_cumsum).astype("uint8")
    return hists_cdf


def clip_histogram(hists, threshold=10.0):
    all_sum = sum(hists)
    threshold_value = all_sum / len(hists) * threshold
    total_extra = sum([h - threshold_value for h in hists if h >= threshold_value])
    mean_extra = total_extra / len(hists)

    clip_hists = [0 for _ in hists]
    for i in range(len(hists)):
        if hists[i] >= threshold_value:
            clip_hists[i] = int(threshold_value + mean_extra)
        else:
            clip_hists[i] = int(hists[i] + mean_extra)

    return clip_hists


def draw_histogram(hists):
    plt.figure()
    plt.bar(range(len(hists)), hists)
    plt.show()


def clahe(img_path, blocks=8, level=256, threshold=10.0):
    img = readimage(img_path)
    img = np.array(img)
    (m, n) = img.shape
    block_m = int(m / blocks)
    block_n = int(n / blocks)


if __name__ == '__main__':
    path = 'car.jpg'
    re = clahe(path)

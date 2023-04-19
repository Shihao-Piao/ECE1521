import numpy as np
from matplotlib import pyplot as plt
import cv2
import metric
from PIL import Image


def readimage(string):
    image = cv2.imread(string)
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


def CLAHE(img, blocks=8, level=256, threshold=10.0):
    (m, n) = img.shape
    block_m = int(m / blocks)
    block_n = int(n / blocks)

    # split small regions and calculate the CDF for each, save to a 2-dim list
    maps = []
    for i in range(blocks):
        row_maps = []
        for j in range(blocks):
            # block border
            si, ei = i * block_m, (i + 1) * block_m
            sj, ej = j * block_n, (j + 1) * block_n

            # block image array
            block_img_arr = img[si: ei, sj: ej]

            # calculate histogram and cdf
            hists = calc_histogram(block_img_arr)
            clip_hists = clip_histogram(hists, threshold=threshold)  # clip histogram
            hists_cdf = calc_histogram_cdf(clip_hists, block_m, block_n, level)

            # save
            row_maps.append(hists_cdf)
        maps.append(row_maps)
    # interpolate every pixel using four nearest mapping functions
    # pay attention to border case
    arr = img.copy()
    for i in range(m):
        for j in range(n):
            r = int((i - block_m / 2) / block_m)  # the row index of the left-up mapping function
            c = int((j - block_n / 2) / block_n)  # the col index of the left-up mapping function

            x1 = (i - (r + 0.5) * block_m) / block_m  # the x-axis distance to the left-up mapping center
            y1 = (j - (c + 0.5) * block_n) / block_n  # the y-axis distance to the left-up mapping center

            lu = 0  # mapping value of the left up cdf
            lb = 0  # left bottom
            ru = 0  # right up
            rb = 0  # right bottom

            # four corners use the nearest mapping directly
            if r < 0 and c < 0:
                arr[i][j] = maps[r + 1][c + 1][img[i][j]]
            elif r < 0 and c >= blocks - 1:
                arr[i][j] = maps[r + 1][c][img[i][j]]
            elif r >= blocks - 1 and c < 0:
                arr[i][j] = maps[r][c + 1][img[i][j]]
            elif r >= blocks - 1 and c >= blocks - 1:
                arr[i][j] = maps[r][c][img[i][j]]
            # four border case using the nearest two mapping : linear interpolate
            elif r < 0 or r >= blocks - 1:
                if r < 0:
                    r = 0
                elif r > blocks - 1:
                    r = blocks - 1
                left = maps[r][c][img[i][j]]
                right = maps[r][c + 1][img[i][j]]
                arr[i][j] = (1 - y1) * left + y1 * right
            elif c < 0 or c >= blocks - 1:
                if c < 0:
                    c = 0
                elif c > blocks - 1:
                    c = blocks - 1
                up = maps[r][c][img[i][j]]
                bottom = maps[r + 1][c][img[i][j]]
                arr[i][j] = (1 - x1) * up + x1 * bottom
            # bilinear interpolate for inner pixels
            else:
                lu = maps[r][c][img[i][j]]
                lb = maps[r + 1][c][img[i][j]]
                ru = maps[r][c + 1][img[i][j]]
                rb = maps[r + 1][c + 1][img[i][j]]
                arr[i][j] = (1 - y1) * ((1 - x1) * lu + x1 * lb) + y1 * ((1 - x1) * ru + x1 * rb)
    arr = arr.astype("uint8")
    return arr
def clahe(path):
    img = readimage(path)
    c = img.shape[2]

    if c == 1:
        re = CLAHE(img)
        return re
    elif c == 3 or c == 4:
        rgb_arr = [None] * 3
        rgb_img = [None] * 3

        for k in range(c):
            rgb_arr[k] = CLAHE(img[:,:,k])
            rgb_img[k] = Image.fromarray(rgb_arr[k])
        img_res = Image.merge("RGB", tuple(rgb_img))
        return img_res


if __name__ == '__main__':
    path = 'image/car.jpg'
    re = clahe(path)

    cv2.imwrite('car_clahe.png',np.array(re))
    plt.imshow(re,cmap='gray')
    plt.show()

    psnr = metric.calculate_psnr(path,'car_clahe.png')
    print(psnr)
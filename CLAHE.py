import numpy as np
from matplotlib import pyplot as plt
import cv2
def readimage(string):
    image = cv2.imread(string, 0)
    imagefinal = cv2.resize(image, (512, 512))
    return imagefinal

def clahe(img_path):
    img =

if __name__ == '__main__':
    
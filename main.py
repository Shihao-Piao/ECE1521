import cv2
import matplotlib.pyplot as plt
import metric
import wiener
import CLAHE
import MSRCR
import numpy as np
'''
#lena.png
path = 'image/lena_noisy.png'
img_origin = cv2.imread(path,0)
img_origin = cv2.resize(img_origin,(512,512))

img_wiener = wiener.wiener_filter(path)
cv2.imwrite('image/lena_wiener.png',np.array(img_wiener))
img_wiener = cv2.imread('image/lena_wiener.png',0)

img_clahe = CLAHE.clahe(path)
cv2.imwrite('image/lena_clahe.png',np.array(img_clahe))
img_clahe = cv2.imread('image/lena_clahe.png',0)

img_msrcr = MSRCR.retinex_MSRCR(path)
cv2.imwrite('image/lena_msrcr.png',np.array(img_msrcr))
img_msrcr = cv2.imread('image/lena_msrcr.png',0)

fig = plt.figure(figsize=(12, 10))
fig.add_subplot(2, 2, 1)
plt.imshow(img_origin, cmap='gray')
plt.title('Noisy Lena')

fig.add_subplot(2, 2, 2)
plt.imshow(img_wiener, cmap='gray')
plt.title('Wiener Filtered')

fig.add_subplot(2, 2, 3)
plt.imshow(img_clahe, cmap='gray')
plt.title('CLAHE')

fig.add_subplot(2, 2, 4)
plt.imshow(img_msrcr, cmap='gray')
plt.title('MSRCR')

plt.savefig('plot/lena.png')
plt.show()

print('Lena:')

psnr_value = metric.calculate_psnr(path,'image/lena_wiener_2.png')
print("wiener filtered PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/lena_wiener_2.png')
print("Entropy of wiener filtered image:", entropy2)

psnr_value = metric.calculate_psnr(path,'image/lena_clahe.png')
print("CLAHE PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/lena_clahe.png')
print("Entropy of CLAHE image:", entropy2)

psnr_value = metric.calculate_psnr(path,'image/lena_msrcr.png')
print("MSRCR PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/lena_msrcr.png')
print("Entropy of MSRCR image:", entropy2)
'''
'''
#car.jpg
path = 'image/car.jpg'
img_origin = cv2.imread(path)
img_origin = cv2.resize(img_origin,(512,512))

img_wiener = wiener.wiener_filter(path)
cv2.imwrite('image/car_wiener.png',np.array(img_wiener))
img_wiener = cv2.imread('image/car_wiener_2.png',0)
img_wiener = cv2.resize(img_wiener,(512,512))

img_clahe = CLAHE.clahe(path)
cv2.imwrite('image/car_clahe.png',np.array(img_clahe))
img_clahe = cv2.imread('image/car_clahe.png',0)

img_msrcr = MSRCR.retinex_MSRCR(path)
cv2.imwrite('image/car_msrcr.png',np.array(img_msrcr))
img_msrcr = cv2.imread('image/car_msrcr.png',0)

fig = plt.figure(figsize=(12, 10))
fig.add_subplot(2, 2, 1)
plt.imshow(img_origin, cmap='gray')
plt.title('Original')

fig.add_subplot(2, 2, 2)
plt.imshow(img_wiener, cmap='gray')
plt.title('Wiener Filtered')

fig.add_subplot(2, 2, 3)
plt.imshow(img_clahe, cmap='gray')
plt.title('CLAHE')

fig.add_subplot(2, 2, 4)
plt.imshow(img_msrcr, cmap='gray')
plt.title('MSRCR')

plt.savefig('plot/car.png')
plt.show()

print('Car:')

psnr_value = metric.calculate_psnr(path,'image/car_wiener_2.png')
print("wiener filtered PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/car_wiener_2.png')
print("Entropy of wiener filtered image:", entropy2)

psnr_value = metric.calculate_psnr(path,'image/car_clahe.png')
print("CLAHE PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/car_clahe.png')
print("Entropy of CLAHE image:", entropy2)

psnr_value = metric.calculate_psnr(path,'image/car_msrcr.png')
print("MSRCR PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/car_msrcr.png')
print("Entropy of MSRCR image:", entropy2)
'''
#medical.png
path = 'image/medical.png'
img_origin = cv2.imread(path)
img_origin = cv2.resize(img_origin,(512,512))

img_wiener = wiener.wiener_filter(path)
cv2.imwrite('image/medical_wiener.png',np.array(img_wiener))
img_wiener = cv2.imread('image/medical_wiener_2.png',0)
img_wiener = cv2.resize(img_wiener,(512,512))

img_clahe = CLAHE.clahe(path)
cv2.imwrite('image/medical_clahe.png',np.array(img_clahe))
img_clahe = cv2.imread('image/medical_clahe.png',0)

img_msrcr = MSRCR.retinex_MSRCR(path)
cv2.imwrite('image/medical_msrcr.png',np.array(img_msrcr))
img_msrcr = cv2.imread('image/medical_msrcr.png',0)

fig = plt.figure(figsize=(12, 10))
fig.add_subplot(2, 2, 1)
plt.imshow(img_origin, cmap='gray')
plt.title('Original')

fig.add_subplot(2, 2, 2)
plt.imshow(img_wiener, cmap='gray')
plt.title('Wiener Filtered')

fig.add_subplot(2, 2, 3)
plt.imshow(img_clahe, cmap='gray')
plt.title('CLAHE')

fig.add_subplot(2, 2, 4)
plt.imshow(img_msrcr, cmap='gray')
plt.title('MSRCR')

plt.savefig('plot/medical.png')
plt.show()

print('Medical:')

psnr_value = metric.calculate_psnr(path,'image/medical_wiener_2.png')
print("wiener filtered PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/medical_wiener_2.png')
print("Entropy of wiener filtered image:", entropy2)

psnr_value = metric.calculate_psnr(path,'image/medical_clahe.png')
print("CLAHE PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/medical_clahe.png')
print("Entropy of CLAHE image:", entropy2)

psnr_value = metric.calculate_psnr(path,'image/medical_msrcr.png')
print("MSRCR PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/medical_msrcr.png')
print("Entropy of MSRCR image:", entropy2)
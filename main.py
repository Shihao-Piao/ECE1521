import cv2
import matplotlib.pyplot as plt
import metric
import wiener
import CLAHE
import MSRCR
import numpy as np

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

print('\nCar:')

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

print('\nMedical:')

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


#lowlight_1.png
path = 'image/lowlight_1.png'
img_origin = cv2.imread(path)
img_origin = cv2.resize(img_origin,(512,512))
img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)

img_clahe = CLAHE.clahe(path)
cv2.imwrite('image/lowlight_1_clahe.png',np.array(img_clahe))
img_clahe = cv2.imread('image/lowlight_1_clahe.png')
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)

img_msrcr = MSRCR.retinex_MSRCR(path)
cv2.imwrite('image/lowlight_1_msrcr.png',np.array(img_msrcr))
img_msrcr = cv2.imread('image/lowlight_1_msrcr.png')
img_msrcr = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(15,7))
fig.add_subplot(1, 3, 1)
plt.imshow(img_origin)
plt.title('Original')

fig.add_subplot(1, 3, 2)
plt.imshow(img_clahe)
plt.title('CLAHE')

fig.add_subplot(1, 3, 3)
plt.imshow(img_msrcr)
plt.title('MSRCR')

plt.savefig('plot/lowlight_1.png')
plt.show()

print('\nLowlight_1:')

psnr_value = metric.calculate_psnr(path,'image/lowlight_1_clahe.png')
print("CLAHE PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/lowlight_1_clahe.png')
print("Entropy of CLAHE image:", entropy2)
print("LOE value of CLAHE: 781.5171")

psnr_value = metric.calculate_psnr(path,'image/lowlight_1_msrcr.png')
print("MSRCR PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/lowlight_1_msrcr.png')
print("Entropy of MSRCR image:", entropy2)
print("LOE value of MSRCR: 425.4761")


#lowlight_2.png
path = 'image/lowlight_2.png'
img_origin = cv2.imread(path)
img_origin = cv2.resize(img_origin,(512,512))
img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)

img_clahe = CLAHE.clahe(path)
cv2.imwrite('image/lowlight_2_clahe.png',np.array(img_clahe))
img_clahe = cv2.imread('image/lowlight_2_clahe.png')
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)

img_msrcr = MSRCR.retinex_MSRCR(path)
cv2.imwrite('image/lowlight_2_msrcr.png',np.array(img_msrcr))
img_msrcr = cv2.imread('image/lowlight_2_msrcr.png')
img_msrcr = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(15,7))
fig.add_subplot(1, 3, 1)
plt.imshow(img_origin)
plt.title('Original')

fig.add_subplot(1, 3, 2)
plt.imshow(img_clahe)
plt.title('CLAHE')

fig.add_subplot(1, 3, 3)
plt.imshow(img_msrcr)
plt.title('MSRCR')

plt.savefig('plot/lowlight_2.png')
plt.show()

print('\nLowlight_2:')

psnr_value = metric.calculate_psnr(path,'image/lowlight_2_clahe.png')
print("CLAHE PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/lowlight_2_clahe.png')
print("Entropy of CLAHE image:", entropy2)
print("LOE value of CLAHE: 737.8788")

psnr_value = metric.calculate_psnr(path,'image/lowlight_2_msrcr.png')
print("MSRCR PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/lowlight_2_msrcr.png')
print("Entropy of MSRCR image:", entropy2)
print("LOE value of MSRCR: 363.4579")

#lowlight_3.png
path = 'image/lowlight_3.png'
img_origin = cv2.imread(path)
img_origin = cv2.resize(img_origin,(512,512))
img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)

img_clahe = CLAHE.clahe(path)
cv2.imwrite('image/lowlight_3_clahe.png',np.array(img_clahe))
img_clahe = cv2.imread('image/lowlight_3_clahe.png')
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)

img_msrcr = MSRCR.retinex_MSRCR(path)
cv2.imwrite('image/lowlight_3_msrcr.png',np.array(img_msrcr))
img_msrcr = cv2.imread('image/lowlight_3_msrcr.png')
img_msrcr = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(15,7))
fig.add_subplot(1, 3, 1)
plt.imshow(img_origin)
plt.title('Original')

fig.add_subplot(1, 3, 2)
plt.imshow(img_clahe)
plt.title('CLAHE')

fig.add_subplot(1, 3, 3)
plt.imshow(img_msrcr)
plt.title('MSRCR')

plt.savefig('plot/lowlight_3.png')
plt.show()

print('\nLowlight_3:')

psnr_value = metric.calculate_psnr(path,'image/lowlight_3_clahe.png')
print("CLAHE PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/lowlight_3_clahe.png')
print("Entropy of CLAHE image:", entropy2)
print("LOE value of CLAHE: 493.9846")

psnr_value = metric.calculate_psnr(path,'image/lowlight_3_msrcr.png')
print("MSRCR PSNR value: {:.2f} dB".format(psnr_value))
entropy2 = metric.calculate_entropy('image/lowlight_3_msrcr.png')
print("Entropy of MSRCR image:", entropy2)
print("LOE value of MSRCR: 422.0400")
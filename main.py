import cv2
import matplotlib.pyplot as plt
import metric

#wiener filter
path1 = 'noisy_lena.png'
path2 = 'wiener_filtered.png'
img1 = cv2.imread(path1,0)
img2 = cv2.imread(path2,0)
display = [img1,img2]
fig = plt.figure()
label = ['noisy','wiener filter']
for i in range(len(display)):
    fig.add_subplot(1, 2, i + 1)
    plt.imshow(display[i], cmap='gray')
    plt.title(label[i])

plt.show()

psnr_value = metric.calculate_psnr(path1,path2)
print("wiener filtered PSNR value: {:.2f} dB".format(psnr_value))

entropy2 = metric.calculate_entropy(path2)
print("Entropy of wiener filtered image:", entropy2)
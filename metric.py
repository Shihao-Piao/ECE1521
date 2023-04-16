import cv2
import numpy as np

def calculate_psnr(img1, img2):
    # Load the two images
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.resize(img2, (512, 512))

    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate the mean squared error (MSE)
    mse = np.mean((img1_gray - img2_gray) ** 2)

    # Calculate the peak signal-to-noise ratio (PSNR)
    max_pixel = 255
    psnr = 10 * np.log10((max_pixel ** 2) / mse)

    return psnr

def calculate_entropy(img):
    """
    Calculate the entropy of an image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.

    Returns:
        float: Entropy value of the image.
    """
    # Load the two images
    image = cv2.imread(img)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram of the grayscale image
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Normalize the histogram
    histogram /= histogram.sum()

    # Calculate the entropy
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))

    return entropy

# Example usage
if __name__ == '__main__':
    img1_path = "lena.png" # Path to the first image
    img2_path = "WechatIMG9.png" # Path to the second image

    psnr_value = calculate_psnr(img1_path, img2_path)
    print("PSNR value: {:.2f} dB".format(psnr_value))

    # Calculate the entropy of the first image
    entropy1 = calculate_entropy('lena.png')

    # Calculate the entropy of the second image
    entropy2 = calculate_entropy('WechatIMG9.png')

    # Print the entropy values
    print("Entropy of image1:", entropy1)
    print("Entropy of image2:", entropy2)

    # Compare the entropy values
    if entropy1 > entropy2:
        print("image1 has higher entropy.")
    elif entropy1 < entropy2:
        print("image2 has higher entropy.")
    else:
        print("Both images have the same entropy.")

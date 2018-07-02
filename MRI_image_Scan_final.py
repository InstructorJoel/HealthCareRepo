import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("C:/Users/jaysh/OneDrive/Documents/Python/Brain_MRI_final.png")
# importing First MRI image
cv2.imshow("Brain with Skull", image)
cv2.waitKey(0)
print(image.shape)
image_1 = cv2.imread("C:/Users/jaysh/OneDrive/Documents/Python/MRI_low_contrast.jpg")
# importing  Second  MRI image
cv2.imshow("Brain with Skull", image_1)
cv2.waitKey(0)
print(image_1.shape)
image_2 = cv2.imread("C:/Users/jaysh/OneDrive/Documents/Python/MRI_brain_new.jpg")
# importing 3rd MRI image
cv2.imshow("MRI Brain", image_2)
cv2.waitKey(0)
print(image_2.shape)


def edge_detection():  # this function detects edges of an MRI image.

    edges = cv2.Canny(image, 70, 70)
    plt.subplot(121), plt.imshow(image_2, cmap='plasma')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def crop_tumor():  # function helps to crop tumor object from MRI scan image.
    tumor = image[120:180, 110:170]
    cv2.imshow("Cropped", tumor)
    cv2.waitKey(0)


def rotate_tumor():  # I have written this function to rotate the tumor image.
    (h, w) = image_2.shape[:2]
    center = (w / 2, h / 2)
    # rotate the image by 180 degrees
    m = cv2.getRotationMatrix2D(center, 180, 2.0)
    rotated = cv2.warpAffine(image_2, m, (w, h))
    cv2.imshow("rotated", rotated)
    cv2.waitKey(0)


def mri_hist_equal():  # Histogram equalization is a method in image processing of contrast adjustment using the
    # image's histogram.

    hist, bins = np.histogram(image_1.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(image_1.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img = cdf[image_1]


# making function call's for desire output.
edge_detection()
crop_tumor()
rotate_tumor()
mri_hist_equal()

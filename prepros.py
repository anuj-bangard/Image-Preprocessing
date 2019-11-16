import cv2
# matplotlib is used for displaying images
import matplotlib.pyplot as plt
# numpy is used for matrix manipulations
import numpy as np


# Read the color image
orig_img = cv2.imread("brain.png",1) # 1 indicates color image
# OpenCV uses BGR while Matplotlib uses RGB format
# Display the color image with matplotlib
plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


#convert image into gray scale
gray_img = cv2.cvtColor( orig_img, cv2.COLOR_BGR2GRAY )
plt.imshow(gray_img,cmap='gray')
plt.axis('off') #same as plt.xticks([]), plt.yticks([])
plt.show()


gaussian_filtered = cv2.GaussianBlur(gray_img, (5, 5),5)
plt.subplot(121), plt.imshow(gray_img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(gaussian_filtered, cmap='gray'),plt.title('Gaussian filter')
plt.xticks([]), plt.yticks([])
plt.show()


median_filtered = cv2.medianBlur(gray_img,5)
plt.subplot(121), plt.imshow(gray_img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(median_filtered, cmap='gray'),plt.title('Median filter')
plt.axis('on')
plt.show()


blur_filtered = cv2.blur(gray_img,(5,5))
plt.subplot(121), plt.imshow(gray_img,cmap='gray'),plt.title('Original')
plt.axis('off')
plt.subplot(122), plt.imshow(blur_filtered, cmap='gray'),plt.title('Mean filter')
#plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.show()


laplacian_filtered = cv2.Laplacian(gray_img,cv2.CV_64F)
plt.subplot(121), plt.imshow(gray_img,cmap='gray'),plt.title('Original')
plt.axis('off')
plt.subplot(122), plt.imshow(laplacian_filtered + gaussian_filtered, cmap='gray'),plt.title('laplacian filter')
#plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.show()
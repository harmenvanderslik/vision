from skimage import data, filters, feature
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

image = data.camera()
mask1 = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
newimage = scipy.ndimage.convolve(image, mask1)
newimage = scipy.ndimage.convolve(newimage, mask1)

# Apply Gaussian blur
sigma = 2  # Adjust sigma as needed for desired blur level
gaussian_blurred_image = filters.gaussian(newimage, sigma=sigma, mode='constant', cval=0)

# Apply Laplacian filter
laplacian_kernel = np.array([[0,  1,  0], 
                             [1, -4,  1], 
                             [0,  1,  0]])
laplacian_image = scipy.ndimage.convolve(newimage, laplacian_kernel)

# Apply Mexican Hat filter
mexican_hat_image = scipy.ndimage.convolve(gaussian_blurred_image, laplacian_kernel)

# Define and apply a high-pass filter
high_pass_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])
high_pass_image = scipy.ndimage.convolve(gaussian_blurred_image, high_pass_kernel)

# Apply Canny edge detection
canny_edges = feature.canny(image, sigma=2)

# Apply Sobel filter
sobel_image = filters.sobel(newimage)

# Apply Scharr filter
scharr_image = filters.scharr(newimage)

# Apply Prewitt filter
prewitt_image = filters.prewitt(newimage)


plt.figure(figsize=(25, 10))

plt.subplot(1, 9, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 9, 2)
plt.imshow(gaussian_blurred_image, cmap='gray')
plt.title('Gaussian Blur')
plt.axis('off')

plt.subplot(1, 9, 3)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian')
plt.axis('off')

plt.subplot(1, 9, 4)
plt.imshow(mexican_hat_image, cmap='gray')
plt.title('Mexican Hat')
plt.axis('off')

plt.subplot(1, 9, 5)
plt.imshow(high_pass_image, cmap='gray')
plt.title('High-Pass')
plt.axis('off')

plt.subplot(1, 9, 9)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge')
plt.axis('off')

plt.subplot(1, 9, 6)
plt.imshow(sobel_image, cmap='gray')
plt.title('Sobel')
plt.axis('off')

plt.subplot(1, 9, 7)
plt.imshow(scharr_image, cmap='gray')
plt.title('Scharr')
plt.axis('off')

plt.subplot(1, 9, 8)
plt.imshow(prewitt_image, cmap='gray')
plt.title('Prewitt')
plt.axis('off')

# Show the plots
plt.show()

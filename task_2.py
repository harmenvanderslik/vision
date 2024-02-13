from skimage import data, filters
from skimage.viewer import ImageViewer
import scipy
from scipy import ndimage

image = data.camera()
viewer = ImageViewer(image)
viewer.show()

mask1=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
newimage=scipy.ndimage.convolve(image, mask1)
newimage=scipy.ndimage.convolve(newimage, mask1)
viewer = ImageViewer(newimage)
viewer.show()
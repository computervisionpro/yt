import cv2
import numpy as np

# Read images
img = cv2.imread("filip-mroz-unsplash.jpg")
obj= cv2.imread("building2.jpg")

# Create an all white mask
mask = 255 * np.ones(obj.shape, obj.dtype)
 
# The location of the center of the src in the dst
height, width, channels = img.shape
print(img.shape)
center = (1200, 1300)
print(f"Width: {center[0]}, Height: {center[1]}")
 

mixed_clone = cv2.seamlessClone(obj, img, mask, center, cv2.MIXED_CLONE)

# Save results
cv2.imwrite("mixed-clone-op.jpg", mixed_clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

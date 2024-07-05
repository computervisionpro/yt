
import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D


# load the image
image = cv2.imread('nemo1.png')
cv2.imshow("original", image)


# convert to rgb
nemo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(nemo.shape)

# get pixel info
pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
print(pixel_colors)
print(pixel_colors.shape)
print()

# pixel values
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

# convert to hsv and flatten
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(image)
hf, sf, vf = h.flatten(), s.flatten(), v.flatten()
print(hf)


# plot
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(hf, sf, vf, facecolors=pixel_colors, marker=".")

axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")

plt.show()






def mask_color():
    
    # define the list of boundaries
    #boundaries = [[100, 110, 100], [120, 255, 255]]
    boundaries = [[0, 110, 100], [10, 255, 255]]


    # create NumPy arrays from the boundaries
    lower = np.array(boundaries[0], dtype = "uint8")
    upper = np.array(boundaries[1] , dtype = "uint8")

    mask = cv2.inRange(image, lower, upper)

    ### show the images
    #cv2.imshow("mask", mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    masked_bgr = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
    cv2.imshow("masked-op", masked_bgr)

    ##
    cv2.waitKey(0)
    cv2.destroyAllWindows()

mask_color()


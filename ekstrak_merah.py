# Import the Packages
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
 
# Load the image using mpimg.imread(). Use a raw string (prefix r) or escape the backslashes.
image = mpimg.imread(
    r'D:\YOLO\dataset\cropped_merah.jpg')
 
# Get the dimensions (width, height, and depth) of the image
w, h, d = tuple(image.shape)
 
# Reshape the image into a 2D array, where each row represents a pixel
pixel = np.reshape(image, (w * h, d))
 
# Import the KMeans class from sklearn.cluster
 
# Set the desired number of colors for the image
n_colors = 1
 
# Create a KMeans model with the specified number of clusters and fit it to the pixels
model = KMeans(n_clusters=n_colors, random_state=42).fit(pixel)
 
# Get the cluster centers (representing colors) from the model
colour_palette = np.uint8(model.cluster_centers_)
 
# Display the color palette as an image
plt.imshow([colour_palette])
 
# Show the plot
plt.show()
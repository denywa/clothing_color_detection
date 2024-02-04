from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = YOLO('cloth.pt')
img_path = './tes_baju_celana.jpg' 

def detect_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    results = model.predict(img)
    # for r in results:
    #     koordinat = r.boxes.xyxy.tolist()[0]
    #     print(koordinat)
    #     for c in r.boxes.cls:
    #         print(model.names[int(c)])
    return results
   

results = detect_image(img_path)

koordinat = []
for r in results:
    koordinat = r.boxes.xyxy.tolist()[0]
    for c in r.boxes.cls:
            print(model.names[int(c)])
print(koordinat[0])
                    
            #  (input_image, output_image, x, y, w, h)             
def crop_image(input_image, output_image, x1, y1, x2, y2):
    img = Image.open(input_image)
    cropped_img = img.crop(box=(x1,y1,x2,y2))
    
    cropped_img.save(output_image)
#koordinat 
x1 = koordinat[0]
y1 = koordinat[1]
x2 = koordinat[2]
y2 = koordinat[3]

output = 'crop.jpg'
crop_image(img_path, output, x1, y1, x2, y2)


# Load the image using mpimg.imread(). Use a raw string (prefix r) or escape the backslashes.
image = mpimg.imread(output)
 
# Get the dimensions (width, height, and depth) of the image
w, h, d = tuple(image.shape)
 
# Reshape the image into a 2D array, where each row represents a pixel
pixel = np.reshape(image, (w * h, d))
 
# Import the KMeans class from sklearn.cluster
 
# Set the desired number of colors for the image
n_colors = 3
 
# Create a KMeans model with the specified number of clusters and fit it to the pixels
model = KMeans(n_clusters=n_colors, random_state=42).fit(pixel)
 
# Get the cluster centers (representing colors) from the model
colour_palette = np.uint8(model.cluster_centers_)
 
# Display the color palette as an image
plt.imshow([colour_palette])
 
# Show the plot
plt.show()




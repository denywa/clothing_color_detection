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


for r in results:
    koordinat = r.boxes.xyxy.tolist()
    for c in r.boxes.cls:
            print(model.names[int(c)])

                            
def crop_image(input_image, output_image, x1, y1, x2, y2):
    img = Image.open(input_image)
    cropped_img = img.crop(box=(x1,y1,x2,y2))
    
    cropped_img.save(output_image)

colour_palettes = []  # Membuat list untuk menyimpan palet warna dari setiap gambar hasil crop

for i in range(len(koordinat)):
    x1 = koordinat[i][0]
    y1 = koordinat[i][1]
    x2 = koordinat[i][2]
    y2 = koordinat[i][3]
    
    output = f'crop{i+1}.jpg'
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
    
    # Simpan palet warna ke dalam daftar
    colour_palettes.append(colour_palette)

# Set up plot untuk menampilkan warna pertama dari setiap palet warna
fig, axs = plt.subplots(1, len(colour_palettes), figsize=(10, 5))

# Tampilkan warna pertama dari setiap palet warna dari setiap gambar hasil crop
for i, palette in enumerate(colour_palettes):
    first_color = palette[0]  # Ambil warna pertama dari palet warna
    
    axs[i].imshow([[first_color]])
    axs[i].axis('off')

plt.tight_layout()
plt.show()

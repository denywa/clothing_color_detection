from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = YOLO('cloth.pt')
img_path = 'D:\YOLO_KTI_2023\CODE BARU\IOU\IOUUUU\long-pants42_jpg.rf.811663af41a03a1f2925f53ed88f098d.jpg' 

def detect_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model.predict(img)
    for r in results:
        koordinat = r.boxes.xywh.tolist()
        classes = [model.names[int(c)] for c in r.boxes.cls]
        confidence_scores = r.boxes.conf.tolist()  # Access confidence scores
        
        #Untuk print isi koordinat dan isi class yang terdeteksi
        for i in range(len(koordinat)):
            print("Class:", classes[i])
            print("Coordinates:", koordinat[i])
            print("Confidence Score:", confidence_scores[i])
            
    return results

results = detect_image(img_path)


# def display_image_with_boxes(image_path, results):
#     img = mpimg.imread(image_path)
    
#     plt.imshow(img)
#     ax = plt.gca()

#     for r in results:
#         koordinat = r.boxes.xyxy.tolist()
#         classes = [model.names[int(c)] for c in r.boxes.cls]

#         for i in range(len(koordinat)):
#             # Ambil koordinat bounding box
#             box = koordinat[i]
            
#             # Ambil kelas dari deteksi
#             class_name = classes[i]

#             # Hitung lebar dan tinggi bounding box
#             width = box[2] - box[0]
#             height = box[3] - box[1]

#             # Buat bounding box dengan koordinat dan label kelasnya
#             rect = plt.Rectangle((box[0], box[1]), width, height, fill=False, color='red')
#             ax.add_patch(rect)
#             ax.text(box[0], box[1], class_name, color='red')

#     plt.axis('off')
#     plt.show()

# display_image_with_boxes(img_path, results)





from PIL import Image

def crop_image(input_image, output_image, x1, y1, x2, y2):
    img = Image.open(input_image)
    cropped_img = img.crop(box=(x1,y1,x2,y2))
    # save gambar
    cropped_img.save(output_image)

#koordinat 
x1 = 67.66709899902344
y1 = 122.62168884277344
x2 = 526.1827392578125
y2 = 392.106689453125
crop_image('bajumerah.jpg', 'tes_crop.jpg', x1, y1, x2, y2)






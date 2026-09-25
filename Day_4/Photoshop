#photoshop
import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# uploading image
uploaded = files.upload()
filename = list(uploaded.keys())[0]

img = cv2.imread(filename)
print(type(img))
print(img.shape)

# showing original image
cv2_imshow(img)

# resizing
img_resize = cv2.resize(img, (400, 256))   # (width, height)
print(img_resize.shape)                    # (256, 400, 3)

# showing resized image
cv2_imshow(img_resize)
#flip
img_resize_flip = cv2.flip(img_resize, 1)
cv2_imshow(img_resize_flip)
#cv2.destroyAllWindows()
#crop
img_crop = img_resize[100:300, 200:500]
cv2_imshow(img_crop)
#cv2.destroyAllWindows()

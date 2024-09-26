import cv2
import easygui
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
from flask import Flask

from google.colab import files
uploaded = files.upload()

# Load the image
import cv2
from matplotlib import pyplot as plt

file_name = list(uploaded.keys())[0]
img = cv2.imread(file_name)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

gray_blur = cv2.medianBlur(gray, 5)
plt.imshow(cv2.cvtColor(gray_blur, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Apply bilateral filter to remove noise and keep edges sharp
color = cv2.bilateralFilter(img, 9, 300, 300)
cartoon = cv2.bitwise_and(color, color, mask=edges)

# Show the final cartoon image
plt.imshow(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the cartoon image
cv2.imwrite('cartoon_image.png', cartoon)

# Download the file (optional)
from google.colab import files
files.download('cartoon_image.png')
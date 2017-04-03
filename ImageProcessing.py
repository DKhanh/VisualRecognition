import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE = './Test/sample2.png'
DIRNAME = './Object/'
IMG_SIZE = 28

img = cv2.imread(IMAGE)

median_blur = cv2.GaussianBlur(img, (3, 3), 0)
grayscaled = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)

gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# detect edges in the image
edged = cv2.Canny(grayscaled, 10, 250)

# To close the grap between lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# Find countours 
_, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

total = 0

idx = 0
for c in contours:
	x,y,w,h = cv2.boundingRect(c)
	if w>IMG_SIZE and h>IMG_SIZE:
		idx+=1
		new_img=img[y:y+h,x:x+w]
		file_name = str(idx) + '.png'
		cv2.imwrite(os.path.join(DIRNAME, file_name), new_img)


cv2.imshow('grayscaled', grayscaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from random import shuffle
from tqdm import tqdm
from datetime import timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pickle
import prettytensor as pt

cell_size = 8
grad_bin = { 0 : 0,
			 20 : 0,
			 40 : 0,
			 60 : 0,
			 80 : 0,
			 100 : 0,
			 120 : 0,
			 140 : 0,
			 160 : 0}

def get_gradient(img) :
    # Calculate the x and y gradients using Sobel operator
    img = np.float32(img)/255
    grad_x = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=1)
    grad_y = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=1)
    magnitude, phase = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    # Combine the two gradients
    grad = cv2.addWeighted((grad_x), 0.5, (grad_y), 0.5, 0)
    return magnitude, phase/2

def extractBlock(block):
	cell_11 = block[0]
	cell_12 = block[1]  #  11 - 12
	cell_21 = block[2]  #  21 - 22 
	cell_22 = block[3]

	top_angle = (cell_11[0][0], cell_11[0][1])
	bot_angle = (cell_22[1][0], cell_22[1][1])

	return top_angle, bot_angle


def segmentImg2Cells(_img):
	cells = {}
	try:
		rows, cols, channels = _img.shape
	except:
		rows, cols = _img.shape
	if rows%cell_size != 0: rows = (rows - rows%cell_size)
	if cols%cell_size != 0: cols = (cols - cols%cell_size)
	img = _img[0:rows, 0:cols]
	# cells = np.ndarray([int(rows/cell_size), int(cols/cell_size)])
	for i in range(0, cols, cell_size):
		for j in range(0, rows, cell_size):
			cells[i/cell_size, j/cell_size] = ([[i, j], [i+cell_size, j+cell_size]])
			# cv2.rectangle(img, (i,j), (i+8,j+8), (255,255,255), 1)

	return img, cells

def segmentImg2Blocks(_img):
	block = {}
	img, cells = segmentImg2Cells(_img)
	try:
		rows, cols, channels = img.shape
	except:
		rows, cols = img.shape
	for i in range(0, int(cols/cell_size-1)):
		for j in range(0, int(rows/cell_size-1)):
			block[i,j] = [cells[i,j],   cells[i,j+1],
					   cells[i+1,j], cells[i+1,j+1]]
			top, bot = extractBlock(block[i,j])
			# cv2.rectangle(img, top, bot, (255-j*20,255-j*8,255-j*6), 1)

	return img, cells, block

def getMidValue(data):
	upper = 0
	lower = 0

	if data == 0: 
		lower = 0
		upper = 20
	elif data == 1: 
		lower = 20
		upper = 40
	elif data == 2: 
		lower = 40
		upper = 60
	elif data == 3: 
		lower = 60
		upper = 80
	elif data == 4: 
		lower = 80
		upper = 100
	elif data == 5: 
		lower = 100
		upper = 120
	elif data == 6: 
		lower = 120
		upper = 140
	elif data == 7: 
		lower = 140
		upper = 160
	elif data == 8: 
		lower = 160
		upper = 0

	return upper, lower

def getPhaseValue(phase):
	idx = int(phase/20)
	r_idx = phase%20
	upper_phase, lower_phase = getMidValue(idx)
	rate = np.float32(1-(r_idx/20))
	return upper_phase, lower_phase, rate

def voteProcess(grad_bin ,upper_phase, lower_phase, upper_magnitude, lower_magnitude):
	grad_bin[upper_phase]  += upper_magnitude
	grad_bin[lower_phase] += lower_magnitude
	return grad_bin

def getMaxValue(grad_bin):
	


def HOGinCells(cell, magnitude, phase):
	grad_bin = { 0 : 0,
				 20 : 0,
				 40 : 0,
				 60 : 0,
				 80 : 0,
				 100 : 0,
				 120 : 0,
				 140 : 0,
				 160 : 0}

	top_col = cell[0][0]
	top_row = cell[0][1]

	bot_col = cell[1][0]
	bot_row = cell[1][1]

	for i in range(top_col, bot_col):
		for j in range(top_row, bot_row):
			upper_phase, lower_phase, rate = getPhaseValue(phase[i,j])
			upper_magnitude = magnitude[i,j]*(1-rate)
			lower_magnitude = magnitude[i,j]*rate
			grad_bin = voteProcess(grad_bin, upper_phase, lower_phase, upper_magnitude, lower_magnitude)

	return grad_bin



img = cv2.imread("./testCifar/street.jpg", cv2.IMREAD_GRAYSCALE)
magnitude, phase = get_gradient(img)
img, cells, block = segmentImg2Blocks(img)

grad_bin = HOGinCells(cells[0,0], magnitude, phase)


print(grad_bin)
print(magnitude[0:8,0:8])
print(phase[0:8,0:8])

# try:
# 	rows, cols, channels = img.shape
# except:
# 	rows, cols = img.shape
# if rows%cell_size != 0: rows = (rows - rows%cell_size)
# if cols%cell_size != 0: cols = (cols - cols%cell_size)

# winSize = (rows,cols)
# blockSize = (16,16)
# blockStride = (8,8)
# cellSize = (cell_size,cell_size)
# nbins = 9
# derivAperture = 1
# winSigma = 4.
# histogramNormType = 0
# L2HysThreshold = 2.0000000000000001e-01
# gammaCorrection = 0
# nlevels = rows
# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
#                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

# winStride = (8,8)
# padding = (8,8)
# locations = ((10,20),)
# hist = hog.compute(img,winStride,padding,locations)

# print(hog.getDescriptorSize())
# print(hist)
# print(block[1,1][0])
# print(magnitude[1,1])
# print(phase[1,1]/2)

# cv2.imshow('test',img)
# cv2.imshow('test_x', grad_x)
# cv2.imshow('test_y', grad_y)
# cv2.imshow('test_', grad)
# cv2.imshow('hist', hist)
# cv2.imshow('magnitude', magnitude)
cv2.imshow('phase', phase*180/360)

cv2.waitKey(0)
import cv2 as cv
import numpy as np
import random
from collections import Counter

class Classifier():
    
	def __init__(self, images):
		
		for i in range(len(images)):
			
			self.img = images[i]
			
			self.img = cv.resize(self.img, self.smart_dimensions(self.img.shape, 800, 800))
			
			self.process(self.img)
			
			
			cv.waitKey(0)
			cv.destroyAllWindows()
	
	
	def process(self, img: cv.Mat):
		
		
		cv.imshow('original', img)
		
		# backSub = cv.createBackgroundSubtractorMOG2()
		# fgMask = backSub.apply(img)
		# cv.imshow('fgmask', fgMask)
		
		bnw = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
		cv.imshow('bnw', bnw)
		
		bg_colour = self.detect_background_colour(img)
		
		# img = cv.threshold(src=img, )
		
		self.create_mask(img, bg_colour)
	
	
	def create_mask(self, image, bg_colour):
		
		img = image.copy()
		
		# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
		
		# create range of colour in hsv
		# 10% above and below current values?
		
		print('bg_colour', bg_colour)
		
		percent = 20/100
		upper_range = [int(min(x*(1+percent), 255)) for x in bg_colour]
		lower_range = [int(max(x*(1-percent), 0)) for x in bg_colour]
		
		mask = cv.inRange(img, np.array(lower_range), np.array(upper_range))
		result = cv.bitwise_and(img, img, mask=mask)
		
		cv.imshow('mask', mask)
		cv.imshow('result', result)
	
	
	# this functions attempts to detect the background colour by randomly sampling points around the images border (5% inward from the edges), 
	# counting the total of all different sampled colours and selecting the mode as the background color
	def detect_background_colour(self, image: cv.Mat):
		
		img = image.copy()
		
		height, width, _channels = img.shape
		
		percent = 0.05
		hBorder = int(percent * height)
		wBorder = int(percent * width)
		
		print(height, width, _channels)
		
		# img[:hBorder, :, :] = 0 # TOP border
		# img[:, :wBorder, :] = 0 # LEFT border
		# img[:, width-wBorder:width, :] = 0 # RIGHT border
		# img[height-hBorder:height, :, :] = 0 # BOTTOM border
		# cv.imshow('border', img)
		
		# increaments tally by one
		def increment_in_dict(dic, key):
			if key in dic:
				dic[key] += 1
			else:
				dic[key] = 1
		
		# keep tally of the colour encountered when randomly sampling points within the 5% border of the image
		counts = {} # key:value --> colour:tally --> [b, g, r]:tally
		for i in range(100):
			
			# colours.append(img[random.randrange(0, hBorder), random.randrange(0, width)]) 				# TOP border
			# colours.append(img[random.randrange(height-hBorder, height), random.randrange(0, width)])	# BOTTOM border
			# colours.append(img[random.randrange(0, height), random.randrange(0, wBorder)])				# LEFT border
			# colours.append(img[random.randrange(0, height), random.randrange(width-wBorder, width)])	# RIGHT border
			
			increment_in_dict(counts, tuple(img[random.randrange(0, hBorder), random.randrange(0, width)]))
			increment_in_dict(counts, tuple(img[random.randrange(height-hBorder, height), random.randrange(0, width)]))
			increment_in_dict(counts, tuple(img[random.randrange(0, height), random.randrange(0, wBorder)]))
			increment_in_dict(counts, tuple(img[random.randrange(0, height), random.randrange(width-wBorder, width)]))
		
		print(counts)
		
		mode_colour = max(counts, key=counts.get)
		print(mode_colour)
		
		return mode_colour
	
	
	def smart_dimensions(self, image_shape, label_width, label_height):
		
		height, width = image_shape[0], image_shape[1]
		
		width_scale = label_width/width
		height_scale = label_height/height
		
		selec = min(width_scale, height_scale)
		
		new_shape = (int(width*selec), int(height*selec))
		
		return new_shape
	


if __name__ == '__main__':
	
	images = []
	images.append(cv.imread('images/img7.jpg'))
	# images.append(cv.imread('images/img2.jpg'))
	
	
	c = Classifier(images)
	
	print('DONE')
import cv2 as cv
import numpy as np
import random
from tkinter import filedialog

# normalising
# thresholding
# masking
# connectedComponents
# countour analysis
# edge detection (canny?)


class Classifier():
    
	def __init__(self, images):
		
		for i in range(len(images)):
			
			self.original = images[i]
			
			self.original = cv.resize(self.original, self.smart_dimensions(self.original.shape, 800, 800))
			
			self.process(self.original)
			
			
			cv.waitKey(0)
			cv.destroyAllWindows()	
	
	
	def process(self, img: cv.Mat):
		
		
		img = self.crop(img)
		
		cv.imshow('original', img)
		
		# backSub = cv.createBackgroundSubtractorMOG2()
		# fgMask = backSub.apply(img)
		# cv.imshow('fgmask', fgMask)
		
		bnw = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
		bnw = cv.blur(bnw, (3, 3))
		cv.imshow('bnw', bnw)
		
		# self.canny_edge(norm)
		self.sobel_edge(bnw)
		
		# bg_colour = self.detect_background_colour(img)
		
		norm = cv.normalize(bnw, None, 0, 300, cv.NORM_MINMAX)
		cv.imshow('normalized', norm)
		
		ret, thresh = cv.threshold(bnw, 125, 255, cv.THRESH_BINARY)
		cv.imshow('thresh', thresh)
		
		self.component_analysis(thresh)
		
		# self.create_mask(img, bg_colour)
	
	
	def sobel_edge(self, image):
		
		img = image.copy()
		
		sx = cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
		sy = cv.Sobel(src=img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
		sxy = cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)

		cv.imshow('sobelx', sx)
		cv.imshow('sobely', sy)
		cv.imshow('sobelxy', sxy)
		
		return sxy
	
	
	def canny_edge(self, image):
		
		img = image.copy()
		
		can = cv.Canny(img, 100, 1000)
		
		cv.imshow('canny', can)
		
	
	
	def component_analysis(self, thresholded_image):
		
		img = thresholded_image.copy()
		
		num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=4)
		
		print('num_labels', num_labels)
		print('output', labels)
		
		sizes = stats[:, -1]
		
		max_label = 1
		max_size = sizes[1]
		for i in range(2, num_labels):
			if sizes[i] > max_size:
				max_label = i
				max_size = sizes[i]
		
		#create black image for the largest component
		component_image = np.zeros(labels.shape)
		component_image= (labels == max_label).astype("uint8")*255
		
		cv.imshow('component', component_image)
		
		
		for i in range(num_labels):
			
			if i == 0:
				text = 'examining component {}/{} (background)'.format(i + 1, num_labels)
			else:
				text = 'examining component {}/{}'.format(i + 1, num_labels)
			
			
			
			x = stats[i, cv.CC_STAT_LEFT]
			y = stats[i, cv.CC_STAT_TOP]
			w = stats[i, cv.CC_STAT_WIDTH]
			h = stats[i, cv.CC_STAT_HEIGHT]
			area = stats[i, cv.CC_STAT_AREA]
			cX, cY = centroids[i]
			
			if area < int(0.05 * img.shape[0] * img.shape[1]) : continue
			
			print('[INFO] {}'.format(text) + ', area=', area)
			
			
			output = self.original.copy()
			cv.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 3)
			cv.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
			
			component_mask = (labels == i).astype('uint8')*255
			
			cv.imshow('output', output)
			cv.imshow('component' + str(i), component_mask)
			cv.waitKey(0)
		
	
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
	
	
	def crop(self, image, crop_percent=0.05):
		
		img = image.copy()
		crop_percent /= 2
		
		print('1st shape=', img.shape)
		
		height, width, _ = img.shape
		
		crop_width = int(crop_percent * width)
		crop_height = int(crop_percent * height)
		
		img = img[ crop_height:height-crop_height , crop_width:width-crop_width, :]
		
		print('2nd shape=', img.shape)
		
		return img
	
	
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
			
			increment_in_dict(counts, tuple(img[random.randrange(0, hBorder), random.randrange(0, width)])) 				# TOP border
			increment_in_dict(counts, tuple(img[random.randrange(height-hBorder, height), random.randrange(0, width)]))		# BOTTOM border
			increment_in_dict(counts, tuple(img[random.randrange(0, height), random.randrange(0, wBorder)]))				# LEFT border
			increment_in_dict(counts, tuple(img[random.randrange(0, height), random.randrange(width-wBorder, width)]))		# RIGHT border
		
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
	# images.append(cv.imread('images/img7.jpg'))
	# # images.append(cv.imread('images/img2.jpg'))
	
	files = filedialog.askopenfilenames()
	for file in files:
		images.append(cv.imread(file))
	
	c = Classifier(images)
	
	print('DONE')
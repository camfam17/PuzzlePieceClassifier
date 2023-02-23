import cv2 as cv
import random
from collections import Counter

class Classifier():
    
	def __init__(self, images):
		
		for i in range(len(images)):
			
			self.img = images[i]
			
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
		
		self.detect_background_colour(img)
		
		# img = cv.threshold(src=img, )

	
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
		
		
		top_points, bottom_points, left_points, right_points = [], [], [], []
		top_colours, bottom_colours, left_colours, right_colours = [], [], [], []
		for i in range(100):
								##### (height, width)
			top_points.append((random.randint(0, hBorder), random.randint(0, width)))
			bottom_points.append((random.randint(height-hBorder, height), random.randint(0, width)))
			left_points.append((random.randint(0, height), random.randint(0, wBorder)))
			right_points.append((random.randint(0, height), random.randint(width-wBorder, width)))
			
			img[random.randrange(0, hBorder), random.randrange(0, width)] = [0, 0, 255]
			img[random.randrange(height-hBorder, height), random.randrange(0, width)] = [0, 255, 0]
			img[random.randrange(0, height), random.randrange(0, wBorder)] = [255, 0, 0]
			img[random.randrange(0, height), random.randrange(width-wBorder, width)] = [0, 122, 122]
			
			top_colours.append(img[random.randrange(0, hBorder), random.randrange(0, width)])
			bottom_colours.append(img[random.randrange(height-hBorder, height), random.randrange(0, width)])
			left_colours.append(img[random.randrange(0, height), random.randrange(0, wBorder)])
			right_colours.append(img[random.randrange(0, height), random.randrange(width-wBorder, width)])
		
		cv.imshow('randoms', img)
		
		
		colours = [*tuple(map(tuple, top_colours)), *tuple(map(tuple, bottom_colours)), *tuple(map(tuple, left_colours)), *tuple(map(tuple, right_colours))]
		counts = Counter(colours)
		print(counts)


if __name__ == '__main__':
	
	images = []
	images.append(cv.imread('images/img1.jpg'))
	# images.append(cv.imread('images/img2.jpg'))
	
	
	c = Classifier(images)
	
	print('DONE')
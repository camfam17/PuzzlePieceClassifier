import cv2 as cv

class Classifier():
    
	def __init__(self, images):
		
		print('hi')
		
		for i in range(1):
			
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
		
		height, width, channels = img.shape
		
		percent = 0.05
		hBorder = int(percent * height)
		wBorder = int(percent * width)
		
		print(height, width, channels)
		
		img[:hBorder, :, :] = 0 # TOP border
		img[:, :wBorder, :] = 0 # LEFT border
		
		cv.imshow('border', img)
		
		pass


if __name__ == '__main__':
    
	img1 = cv.imread('images/img1.jpg')
	
	images = [img1]
	
	c = Classifier(images)
	
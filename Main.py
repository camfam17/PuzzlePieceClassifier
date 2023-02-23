import cv2 as cv

class Classifier():
    
	def __init__(self, images):
		
		print('hi')
		
		for i in range(1):
			
			self.img = images[i]
			
			cv.imshow('img1', self.img)


if __name__ == '__main__':
    
	img1 = cv.imread('images/img1.jpg')
	
	images = [img1]
	
	c = Classifier(images)
	
	cv.waitKey(0)
	cv.destroyAllWindows()
import linear_model
import processImages
import numpy as np
import cv2


class useMlModel:
	def __init__(self,imagePath ="trainingData/trainingImages/image" ,labelPath = "trainingData/trainingLabel.txt"):
		self.imagePath = imagePath
		self.labelPath = labelPath
		self.sigma = 1.0
		#self.model = linear_model.LeastSquaresRBF(self.sigma)
		self.numberOfExamples = self.getNumberOfTrainingExamples()

	def getNumberOfTrainingExamples(self):
		f = open(self.labelPath, 'r')
		line = f.readline()
		numExamples = 0
		while line:
			numExamples += 1
			line = f.readline()

		f.close()
		return numExamples

	'''
		Code from Professor Mike Gelbart, UBC
		Get the best sigma using validation error
	'''
	def getBestSigma(self,X,y):
		 # get the number of rows(n) and columns(d)
		(n,d) = X.shape

		# Split training data into a training and a validation set
		Xtrain = X[0:n//2]
		ytrain = y[0:n//2]
		Xvalid = X[n//2: n]
		yvalid = y[n//2: n]

		# Find best value of RBF kernel parameter,
		# training on the train set and validating on the validation set

		minErr = np.inf
		for s in range(0,16):
			print "s, ", s
			sigma = 2 ** s

			model = linear_model.LeastSquaresRBF(sigma)
			model.fit(Xtrain,ytrain)

			# Compute the error on the validation set
			yhat = model.predict(Xvalid)
			validError = np.sum((yhat - yvalid)**2)/ (n//2)

			if validError < minErr:
				minErr = validError
				bestSigma = sigma

		return bestSigma

	def train(self):
		xTrain = processImages.convertImageToArray(self.numberOfExamples, self.imagePath)
		yTrain = processImages.convertLabelToArray(self.numberOfExamples, self.labelPath)
		bestSigma = self.getBestSigma(xTrain,yTrain)
		self.model = linear_model.LeastSquaresRBF(bestSigma)
		self.model.fit(xTrain,yTrain)

	def predict(self,testX):
		yhat = self.model.predict(testX)
		return yhat[0]

	def testOnTrainingData(self):
		Xtrain = processImages.convertImageToArray(self.numberOfExamples, self.imagePath)
		ytrain = processImages.convertLabelToArray(self.numberOfExamples, self.labelPath)		
		#yhat = self.model.predict(Xtrain)
		
		img = cv2.imread("testPredictionImage.png")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = gray/255.0
		XtrainNew = np.reshape(gray,(1,gray.shape[1]*gray.shape[0]))
		#XtrainNew = np.zeros((1,Xtrain.shape[1]))
		#XtrainNew[0,:] = Xtrain[24]
		print "nonzero XtrainNew ", XtrainNew[XtrainNew!=0]
		yhat = self.predict(XtrainNew)

		print "yhat ", yhat
		#trainingError = np.sum((yhat - ytrain[24])**2)
		#print "trainingError", trainingError


if __name__ == "__main__":
	modelTest = useMlModel()
	modelTest.train()
	modelTest.testOnTrainingData()


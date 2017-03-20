import linear_model
import processImages
import numpy as np
import cv2



class useMlModel:
	def __init__(self,imagePath ="trainingData/trainingFeature.txt" ,labelPath = "trainingData/trainingLabel.txt"):
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
		Get the best sigma using cross validation
	'''
	def getBestSigma(self,X,y):
		print "Finding best sigma "
		 # get the number of rows(n) and columns(d)
		(n,d) = X.shape

		# Split training data into a training and a validation set
		#Xtrain = X[0:n//2]
		#ytrain = y[0:n//2]
		#Xvalid = X[n//2: n]
		#yvalid = y[n//2: n]

		# Find best value of RBF kernel parameter,
		# training on the train set and validating on the validation set
		numFolds = 10
		finalSigma = 0
		minErr = np.inf
		for s in range(0,16):
			print "s ", s
			sigma = 2 ** s
			startFold = 0
			sumErrors = 0
			for nF in range(numFolds):
				Xvalid = X[startFold: startFold+n//numFolds]
				yvalid = y[startFold: startFold+n//numFolds]

				Xtrain = X[0:startFold]
				ytrain = y[0:startFold]
				np.append(Xtrain,X[startFold+n//numFolds:n],axis=0)
				np.append(ytrain,y[startFold+n//numFolds:n],axis=0)

				startFold = startFold+n//numFolds
			

				# Train on the training set
				model = linear_model.LeastSquaresRBF(sigma)
				model.fit(Xtrain,ytrain)

				# Compute the error on the validation set
				yhat = model.predict(Xvalid)
				validError = np.sum((yhat - yvalid)**2)/ (n//2)
				sumErrors += validError
				print("Error with sigma = {:e} = {}".format( sigma ,validError))

			validError = sumErrors/numFolds
			# Keep track of the lowest validation error
			if validError < minErr:
				minErr = validError
				bestSigma = sigma
		return bestSigma

	def train(self):
		print "Training"
		#xTrain = processImages.convertImageToArray(self.numberOfExamples, self.imagePath)
		xTrain = processImages.constructXFromTargetFocusLocations(self.numberOfExamples, self.imagePath)
		yTrain = processImages.convertLabelToArray(self.numberOfExamples, self.labelPath)
		bestSigma = self.getBestSigma(xTrain,yTrain)
		self.model = linear_model.LeastSquaresRBF(bestSigma)
		self.model.fit(xTrain,yTrain)

	def predict(self,testX):
		testFeature = np.zeros((1,4))
		testFeature[0,:] = testX
		yhat = self.model.predict(testFeature)
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


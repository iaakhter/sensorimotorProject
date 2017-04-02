#import linear_model
import processImages
import numpy as np
import cv2
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


class useMlModel:
	def __init__(self,imagePath ="trainingData/trainingFeatureXY.txt" ,labelPath = "trainingData/trainingLabelXY.txt"):
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


	def train(self):
		print "Training"
		#xTrain = processImages.convertImageToArray(self.numberOfExamples, self.imagePath)
		xTrain = processImages.constructXFromTargetFocusLocations(self.numberOfExamples, 4,self.imagePath)
		yTrain = processImages.convertLabelToArray(self.numberOfExamples, 2,self.labelPath)
		yTrain = np.reshape(yTrain,(xTrain.shape[0],2))
		self.model = MLPRegressor(hidden_layer_sizes=(100,),alpha=1.0)
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


#import linear_model
import processImages
import numpy as np
import cv2
import tensorflow as tf
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor



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
		print ("shape of xTrain: "), xTrain.shape
		#bestSigma = self.getBestSigma(xTrain,yTrain)
		#self.model = linear_model.LeastSquaresRBF(bestSigma)
		#self.model.fit(xTrain,yTrain)
		yTrain = np.reshape(yTrain,(xTrain.shape[0]))
		#self.model = linear_model.LinearRegression()
		#self.model.fit(xTrain,yTrain)
		self.model = MLPRegressor(hidden_layer_sizes=(70,),alpha=0.01)
		self.model.fit(xTrain,yTrain)


	def predict(self,testX):
		testFeature = np.zeros((1,2))
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

	def trainTensor(self):
		#code from https://github.com/nethsix/gentle_tensorflow/blob/master/code/linear_regression_multi_feature_using_mini_batch_with_tensorboard.py
		datapoint_size = 1000
		batch_size = 1000
		steps = 10000
		learn_rate = 0.001
		log_file = "/tmp/feature_2_batch_1000"

		#Model linear regression y = Wx + b
		x = tf.placeholder(tf.float32, [None, 2], name="x")
		W = tf.Variable(tf.zeros([2,1]), name="W")
		b = tf.Variable(tf.zeros([1]), name="b")
		with tf.name_scope("Wx_b") as scope:
			product = tf.matmul(x,W)
			y = product + b

		# Add summary ops to collect data
		W_hist = tf.summary.histogram("weights", W)
		b_hist = tf.summary.histogram("biases", b)
		y_hist = tf.summary.histogram("y", y)	

		y_ = tf.placeholder(tf.float32, [None, 1])

		# Cost function sum((y_-y)**2)
		with tf.name_scope("cost") as scope:
  			cost = tf.reduce_mean(tf.square(y_-y))
  			cost_sum = tf.summary.scalar("cost", cost)

		# Training using Gradient Descent to minimize cost
		with tf.name_scope("train") as scope:
  			train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

  		
  		xTrain = processImages.constructXFromTargetFocusLocations(datapoint_size, self.imagePath)
  		xTrain = np.array(xTrain)

  		yTrain = processImages.convertLabelToArray(datapoint_size, self.labelPath)
  		yTrain = np.reshape(yTrain,(xTrain.shape[0]))

  		sess = tf.Session()

  		# Merge all the summaries and write them out to /tmp/mnist_logs
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(log_file, sess.graph_def)

		# init = tf.initialize_all_variables()
		init = tf.global_variables_initializer()
		sess.run(init)

		for i in range(steps):
  			if datapoint_size == batch_size:
  				batch_start_idx = 0
  			elif datapoint_size < batch_size:
  				raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
  			else:
  				batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
  			batch_end_idx = batch_start_idx + batch_size
  			batch_xs = xTrain[batch_start_idx:batch_end_idx]
  			batch_ys = yTrain[batch_start_idx:batch_end_idx]
  			xs = np.array(batch_xs)
  			ys = np.array(batch_ys)
  			all_feed = { x: xs, y_: ys }
  			# Record summary data, and the accuracy every 10 steps
  			if i % 10 == 0:
  				# result = sess.run(merged, feed_dict=all_feed)
  				# writer.add_summary(result, i)
  				print("recording summary data removed")
  			else:
  				feed = { x: xs, y_: ys }
  				sess.run(train_step, feed_dict=feed)
  			# print("After %d iteration:" % i)
  			# print("W: %s" % sess.run(W))
  			# print("b: %f" % sess.run(b))
  			# print("cost: %f" % sess.run(cost, feed_dict=all_feed))


if __name__ == "__main__":
	modelTest = useMlModel()
	modelTest.train()
	modelTest.trainTensor()
	# modelTest.testOnTrainingData()


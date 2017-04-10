from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import processImages 
import numpy as np

# the data, shuffled and split between train and test sets
# (X_train, y_train_cat), (X_test, y_test_cat) = mnist.load_data()
class kerasNet:
    def __init__(self):
        # self.imagePath ="trainingData/trainingFeatureXY.txt"
        # self.labelPath = "trainingData/trainingLabelXY.txt"
        # self.imagePath ="trainingData/trainingGaussianFeatureXY.txt"
        # self.labelPath = "trainingData/trainingGaussianLabelXY.txt"

        self.numOfExamples = self.getNumberOfTrainingExamples()


        self.numOfFeatures = 4
        self.numOfLabels = 2

    def train(self):
        print "Training keras with ", self.numOfExamples, " examples "
        xTrain = processImages.constructXFromTargetFocusLocations(self.numOfExamples, self.numOfFeatures, self.imagePath)
        yTrain = processImages.convertLabelToArray(self.numOfExamples, self.numOfLabels, self.labelPath)


        self.model = Sequential()
        self.model.add(Dense(200, input_shape=(4,), activation='relu', use_bias=True))
        # self.model.add(Dropout(0.8))
        # self.model.add(Dense(100, activation='relu'))
        # self.model.add(Dropout(0.8))
        # self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(2))

        self.model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['mse'])

        history = self.model.fit(xTrain, yTrain,
                            batch_size=128, 
                            epochs=500,
                            verbose=0,
                            validation_split=0.8)

        score = self.model.evaluate(xTrain, yTrain, verbose=0)
        print('Test accuracy:', score[1])

    def predict(self, xTest):
        xTest = np.reshape(xTest, (1,4))
        yhat = self.model.predict(xTest)
        return yhat

    def getNumberOfTrainingExamples(self):
        f = open(self.labelPath, 'r')
        line = f.readline()
        numExamples = 0
        while line:
            numExamples += 1
            line = f.readline()

        f.close()
        return numExamples

if __name__ == "__main__":
    model = kerasNet()
    model.train()



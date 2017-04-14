from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import processImages 
import numpy as np

# the data, shuffled and split between train and test sets
# (X_train, y_train_cat), (X_test, y_test_cat) = mnist.load_data()
class kerasConvNet:
    def __init__(self):
        self.imagePath ="trainingData/resizedImages/image"
        self.labelPath = "trainingData/trainingLabelXY.txt"

        self.numOfExamples = self.getNumberOfTrainingExamples()
        print "number of training examples: ", self.numOfExamples
        self.numOfFeatures = 4
        self.numOfLabels = 2
        self.img_dim = (50,50)

    def train(self):
        print "Training convolutional net keras"
        xTrain = processImages.convertImageToArrayColor(self.numOfExamples, self.imagePath)
        xTrain = np.reshape(xTrain, (self.numOfExamples,50,50,3))

        yTrain = processImages.convertLabelToArray(self.numOfExamples, self.numOfLabels, self.labelPath)
        # yTrain = np.reshape(yTrain, (self.numOfExamples, 2, 1, 1))


        self.model = Sequential()
        self.model.add(Convolution2D(32, (3, 3), input_shape=(50,50,3), activation='relu', use_bias=True))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Convolution2D(100, (3, 3), activation='relu'))
        #self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(32,activation='relu'))
        self.model.add(Dense(2))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])


        history = self.model.fit(xTrain, yTrain,
                            batch_size=64, 
                            epochs=200,
                            verbose=1,
                            validation_split=0.1)

        score = self.model.evaluate(xTrain, yTrain, verbose=0)
        print('Test accuracy:', score[1])
        self.model.save('myKerasConvNet.h5')

    def predict(self, xTest):
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
    model = kerasConvNet()
    model.train()
    Xtest = processImages.convertImageToArrayColor (15, "testData/resizedImages/image")
    Xtest = np.reshape(Xtest, (15, 50, 50, 3))
    prediction = model.predict(Xtest)
    print "prediction: ", prediction



import linear_model
import processImages

sigma = 1.0
numberOfExamples = 234
imagePath = "trainingData/trainingImages/image"
labelPath = "trainingData/trainingLabel.txt"
Xtrain = processImages.convertImageToArray(numberOfExamples, imagePath)
ytrain = processImages.convertLabelToArray(numberOfExamples, labelPath)
model = linear_model.LeastSquaresRBF(sigma)
model.fit(Xtrain,ytrain)


yhat = model.predict(Xtrain)
trainingError = np.sum((yhat - ytest)**2)

print trainingError
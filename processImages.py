import numpy as np
import cv2

def convertImageToArray (numberOfExamples, imagePath):
    #Get dimensions
    firstImageName = imagePath + str(0) +'.png'
    img = cv2.imread(firstImageName)
    [n,d,t] = img.shape
    
    
    X = np.zeros((numberOfExamples, n*d))

    
    for i in range(numberOfExamples):
        imageName = imagePath + str(i) +'.png'
        img = cv2.imread(imageName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X[i] = np.reshape(gray, (1, n*d))
        
    return X
    
def convertLabelToArray (numberOfExamples, labelPath):
    #Deal with labels
    trainingLabelsFile = open(labelPath, 'r')
    
    y = np.zeros((numberOfExamples,1))
    
    for i in range(numberOfExamples):
        currentLabel = float(trainingLabelsFile.readline())
        y[i] = currentLabel
    
    
    trainingLabelsFile.close()
    return y


#test run
#X = convertImageToArray(234, 'trainingData/trainingImages/image')
#y = convertLabelToArray(234, 'trainingData/trainingLabel.txt')
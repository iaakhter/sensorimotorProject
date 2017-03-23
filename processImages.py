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
        gray = gray/255.0
        X[i] = np.reshape(gray, (1, n*d))
        
    return X

def constructXFromTargetFocusLocations(numberOfExamples, filePath):
    #Get dimensions
    trainingFeaturesFile = open(filePath, 'r')
    
    X = np.zeros((numberOfExamples,2))
    
    for i in range(numberOfExamples):
        orientationList = trainingFeaturesFile.readline()
        orientationList = orientationList.split(" ")
        for j in range(2):
            X[i,j] = float(orientationList[j])
    
    
    trainingFeaturesFile.close()
    return X

    
def convertLabelToArray (numberOfExamples, labelPath):
    #Deal with labels
    trainingLabelsFile = open(labelPath, 'r')
    
    y = np.zeros((numberOfExamples,1))
    
    for i in range(numberOfExamples):
        currentLabel = float(trainingLabelsFile.readline())
        y[i] = float(currentLabel)
    
    
    trainingLabelsFile.close()
    return y


if __name__ == "__main__":
   #test run
    #X = convertImageToArray(234, 'trainingData/trainingImages/image')
    #y = convertLabelToArray(234, 'trainingData/trainingLabel.txt')
    X = constructXFromTargetFocusLocations(1,"trainingData/trainingFeature.txt")
    print X
import numpy as np
import cv2
from PIL import Image

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

def convertImageToArrayColor (numberOfExamples, imagePath):
    #Get dimensions
    firstImageName = imagePath + str(0) +'.png'
    img = cv2.imread(firstImageName)
    [n,d,t] = img.shape
    
    
    X = np.zeros((numberOfExamples, n*d*t))

    
    for i in range(numberOfExamples):
        imageName = imagePath + str(i) +'.png'
        img = cv2.imread(imageName)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = gray/255.0
        X[i] = np.reshape(img, (1, n*d*t))
        
    return X

def constructXFromTargetFocusLocations(numberOfExamples, ndimensions, filePath):
    #Get dimensions
    trainingFeaturesFile = open(filePath, 'r')
    
    X = np.zeros((numberOfExamples,ndimensions))
    
    for i in range(numberOfExamples):
        orientationList = trainingFeaturesFile.readline()
        orientationList = orientationList.split(" ")
        for j in range(ndimensions):
            X[i,j] = float(orientationList[j])
    
    
    trainingFeaturesFile.close()
    return X

    
def convertLabelToArray (numberOfExamples, ndimensions, labelPath):
    #Deal with labels
    trainingLabelsFile = open(labelPath, 'r')
    
    y = np.zeros((numberOfExamples, ndimensions))
    
    for i in range(numberOfExamples):
        currentLabelList = trainingLabelsFile.readline()
        currentLabelList = currentLabelList.split(" ")
        for j in range(ndimensions):
            y[i,j] = float(currentLabelList[j])
    
    
    trainingLabelsFile.close()
    return y

def resizeImages (numberOfExamples, filePath, savePath):
    for i in range(numberOfExamples):
        currImage = filePath + str(i) +'.png'
        img = Image.open(currImage)

        resizedImage = img.resize((50,50), Image.BICUBIC)

        imageName = savePath + str(i) + ".png"
        resizedImage.save(imageName, 'PNG')

def standardizeCols(M, *args):
        stdM = np.zeros(M.shape, dtype =np.float32)
        if (len(args) > 0):
            mu =  args[0]
            std_dev = args[1]
        else:
        #if mu and sigma are omitted, compute from M
            mu = np.mean(M, axis=0)
            std_dev = np.std(M, axis=0)

        stdM = (M - mu) / std_dev

        return stdM, mu, std_dev
if __name__ == "__main__":
   #test run
    #X = convertImageToArray(234, 'trainingData/trainingImages/image')
    # y = convertLabelToArray(1, 4, 'trainingData/trainingLabelXY.txt')
    # X = constructXFromTargetFocusLocations(1,4,"trainingData/trainingFeatureXY.txt")
    # resizeImages (2105,"trainingData/trainingImagesXY/image", "trainingData/resizedImages/image")
    resizeImages (329,"testData/testImages/image", "testData/resizedImages/image")
    # X = convertImageToArrayColor(1, "trainingData/resizedImages/image")
    # stdy, muTrain, sigmaTrain = standardizeCols(y)
    # print stdy, muTrain, sigmaTrain
    # print X.shape
    # print np.max(X)
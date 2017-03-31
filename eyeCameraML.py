from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import *
from math import *
from time import time
from quaiaoptican import *

import tensorflow as tf
import cv2
import useMl
import useTensor
import model_example
import processImages
import train_example
# References:
#         https://noobtuts.com/python/opengl-introduction
#         https://learnopengl.com/#!Getting-started/Camera
#         https://gist.github.com/strife25/803118

class eyeCamera:

    def __init__(self):
        self.width, self.height = 500, 500
        self.targetX, self.targetY, self.targetZ = 0, 0, 0
        self.eyeInitOrient = array([[0], [0], [0]])
        self.innervSignal = array([[0.00000001],[0],[0]])
        self.initCameraRotAxis = array([0.0,0.0,0.0])
        self.initCameraRotAngle = 0.0
        self.cameraRotAxis = array([0.0,0.0,0.0])
        self.cameraRotAngle = 0.0
        self.targetChanged = False
        self.predictInnerv = False
        self.mlModel = useMl.useMlModel()
        self.mlModel.train()
        # self.tensorModel = useTensor.useMlModel()
        # self.tensorModel.trainTensor()
        tensorModel = train_example.trainCNN()
        self.traningMean = tensorModel.muYTrain
        self.traningStd = tensorModel.stdYTrain
        #start session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        #Load Model 
        saver = tf.train.Saver()
        saver.restore(self.sess, "./model.ckpt")
    
    def setUpCamera(self,cameraPosition,cameraTarget,cameraUp,
                    perceivedTargetWidth,perceivedTargetHeight):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glPushMatrix()
        glTranslate(0.0,0.0,1.0)
        # rotate the camera by the angle
        glRotate(self.cameraRotAngle,self.cameraRotAxis[0], self.cameraRotAxis[1], self.cameraRotAxis[2])
        
        # since opengl forgets previous rotations, first rotate the camera by previous rotation
        glRotate(self.initCameraRotAngle,self.initCameraRotAxis[0], self.initCameraRotAxis[1], self.initCameraRotAxis[2])
        glTranslate(0.0,0.0,-1.0)
        gluLookAt(cameraPosition[0],cameraPosition[1],cameraPosition[2],
                  cameraTarget[0],cameraTarget[1],cameraTarget[2],
                  cameraUp[0],cameraUp[1],cameraUp[2])

        # let's look at the target that the camera perceives. This indicates
        # the region that is in focus by the camera
        self.drawPositionOffocus(cameraPosition,perceivedTargetWidth,perceivedTargetHeight)
        glPopMatrix()

    def drawRectangle(self,x, y, z, width, height,color):
        glColor3f(color[0],color[1],color[2])
        glBegin(GL_POLYGON)
        glVertex3f(x-width/2.0, y-height/2.0, z)                               
        glVertex3f(x+width/2.0, y-height/2.0, z)                      
        glVertex3f(x+width/2.0, y+height/2.0, z)            
        glVertex3f(x-width/2.0, y+height/2.0, z)
        glEnd()

    def drawLine(self,x1, y1, z1, x2, y2, z2,color):
        glColor3f(color[0],color[1],color[2])
        glBegin(GL_LINES)
        glVertex3f(x1,y1,z1)
        glVertex3f(x2,y2,z2)
        glEnd()

    def drawTarget(self,x,y,z,width,height):
        color = (0.0,0.0,1.0)
        self.drawRectangle(x,y,z,width,height,color)

        # Draw the x,y and z axes
        '''color = (1.0,0.0,0.0)
        self.drawLine(x,y,z,x+0.5,y,z, color)
        color = (0.0,1.0,0.0)
        self.drawLine(x,y,z,x,y+0.5,z, color)
        color = (0.0,0.0,1.0)
        self.drawLine(x,y,z,x,y,z+0.5, color)'''


    # Draw a square representing the focus of the eye (along the optical axis)
    def drawPositionOffocus(self, cameraPosition,perceivedTargetWidth,perceivedTargetHeight):
        mvmatrix = glGetDoublev (GL_MODELVIEW_MATRIX);
        visualAxis = array([mvmatrix[2][0], mvmatrix[2][1], mvmatrix[2][2]])
        pointOffocus = cameraPosition - visualAxis
        #print "visualAxis: ", visualAxis
        #print "pointOffocus: ", pointOffocus
        color = (1.0,0.0,0.0)
        if(pointOffocus[2] < 0.003):
            pointOffocus[2] = 0.005
        self.drawRectangle(pointOffocus[0],pointOffocus[1],pointOffocus[2],perceivedTargetWidth,perceivedTargetHeight,color)
        


    def keyPressed(self,*args):
        if(args[0] == 'w'):
            self.targetChanged = True
            self.targetY += 0.1
        elif(args[0] == 's'):
            self.targetChanged = True
            self.targetY -= 0.1
        elif(args[0] == 'd'):
            self.targetChanged = True
            self.targetX += 0.5
        elif(args[0] == 'a'):
            self.targetChanged = True
            self.targetX -= 0.5

    def OnMouseClick(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            print "click on (x,y): ", x, y
            self.targetChanged = True
            self.targetX = (x / 250.0) - 1
            self.targetY = -(y / 250.0) + 1

    def determineTargetCameraFocusPosition(self):
        #Get the screen coordinates for blue and red objects
        pixels = glReadPixels(0.0,0.0,self.width,self.height,format=GL_RGB,type=GL_FLOAT)
        #print "pixels"
        indicesB = where(pixels[:,:,2]==1.0)

        if len(indicesB[0]) > 0:
            startBluey = min(indicesB[0])
            endBluey = max(indicesB[0])
            startBluex = min(indicesB[1])
            endBluex = max(indicesB[1])
            centerBluex = (startBluex + endBluex)/2.0
            centerBluey = (startBluey + endBluey)/2.0

        indicesR = where(pixels[:,:,0]==1.0)

        if len(indicesR[0]) > 0:
            startRedy = min(indicesR[0])
            endRedy = max(indicesR[0])
            startRedx = min(indicesR[1])
            endRedx = max(indicesR[1])
            centerRedx = (startRedx + endRedx)/2.0
            centerRedy = (startRedy + endRedy)/2.0

        #print "centerBluex ", centerBluey
        #print "centerRedx ", centerRedx

        if len(indicesB[0]) > 0 and len(indicesR[0]):
            # return [centerBluex, centerRedx]
            return [centerBluex, centerBluey, centerRedx, centerRedy]
        else:
            return []

    def getTestExample(self):
        pixels = glReadPixels(0.0,0.0,self.width,self.height,format=GL_RGB,type=GL_FLOAT)
        pixels = pixels*255
        imageName = "testData/testImage0.png"
        cv2.imwrite(imageName,pixels)
        resized_image = processImages.resizeImages (1, "testData/testImage", "testData/resizedImage")
        Xtest = processImages.convertImageToArrayColor (1, "testData/resizedImage")
        Xtest = np.reshape(Xtest, (1, 50, 50, 3))
        return Xtest


    # taken from http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToEuler/
    def convertAxisAngleToEuler(self,rotAxis,rotAngle):
        s = sin(rotAngle)
        c = cos(rotAngle)
        t = 1-c
        x,y,z = rotAxis[0], rotAxis[1], rotAxis[2]
        #  if axis is not already normalised then uncomment this
        # double magnitude = Math.sqrt(x*x + y*y + z*z);
        # if (magnitude==0) throw error;
        # x /= magnitude;
        # y /= magnitude;
        # z /= magnitude;
        if ((x*y*t + z*s) > 0.998): # north pole singularity detected
            heading = 2*atan2(x*sin(rotAngle/2),cos(rotAngle/2))
            attitude = pi/2
            bank = 0
        elif ((x*y*t + z*s) < -0.998): # south pole singularity detected
            heading = -2*atan2(x*sin(rotAngle/2),cos(rotAngle/2))
            attitude = -PI/2
            bank = 0
        else:
            heading = atan2(y * s- x * z * t , 1 - (y*y+ z*z ) * t);
            attitude = asin(x * y * t + z * s) ;
            bank = atan2(x * s - y * z * t , 1 - (x*x + z*z) * t);
        # (myVersion = author's versoin : x = bank, roll; y = heading, yaw; z = attitude, pitch)
        return array([[bank],[heading],[attitude]])

    
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToAngle/
    def convertEulerToAxisAngle(self, heading, attitude, bank):
        # Assuming the angles are in radians.
        c1 = cos(heading/2);
        s1 = sin(heading/2);
        c2 = cos(attitude/2);
        s2 = sin(attitude/2);
        c3 = cos(bank/2);
        s3 = sin(bank/2);
        c1c2 = c1*c2;
        s1s2 = s1*s2;
        w =c1c2*c3 - s1s2*s3;
        x =c1c2*s3 + s1s2*c3;
        y =s1*c2*c3 + c1*s2*s3;
        z =c1*s2*c3 - s1*c2*s3;
        angle = 2 * acos(w);
        norm = x*x+y*y+z*z;
        if norm >= 0.001:
            norm = sqrt(norm);
            x /= (1.0*norm);
            y /= (1.0*norm);
            z /= (1.0*norm);
        return [angle,x,y,z]


    #callback function for opengl
    def setUpSystem(self):
        cameraPosition = array([0.0,0.0,1.0])
        cameraTarget = array([0.0,0.0,0.0])
        cameraUp = array([0.0,1.0,0.0])

        perceivedTargetWidth = 0.30
        perceivedTargetHeight = 0.30

        self.setUpCamera(cameraPosition,cameraTarget,cameraUp,
                        perceivedTargetWidth,perceivedTargetHeight)

        #update the initial camera rotation and angle to be the current ones
        self.initCameraRotAxis = self.cameraRotAxis
        self.initCameraRotAngle = self.cameraRotAngle
        
        targetWidth = 0.25
        targetHeight = 0.25
        self.drawTarget(self.targetX,self.targetY,self.targetZ,targetWidth,targetHeight)
        
        if self.predictInnerv:
            self.predictInnerv = False
            featureVector = self.determineTargetCameraFocusPosition()

            predictedInnervXY = self.mlModel.predict(featureVector)

            # xTest = self.getTestExample() 
            # xTest = np.reshape(featureVector, (1,1,2,1))
            xTest = np.reshape(featureVector, (1,1,4,1))
            test_prediction = model_example.y.eval(session=self.sess, feed_dict={model_example.x: xTest, model_example.keep_prob: 1.0})

            # print "predicted sklearn ", predictedInnervY

            predictedCNN = test_prediction*self.traningStd + self.traningMean

            print "predicted tensor: ", predictedCNN

            # only rotate if the boxes will remain in the window upon rotation
            #if self.eyeInitOrient[1]*predictedInnervY >= -5000 and self.eyeInitOrient[1]*predictedInnervY <= 5000:
            # for x movements
            self.innervSignal = array([[predictedInnervXY[0]],[predictedInnervXY[1]],[0]])
            # self.innervSignal = array([[0.00000001],[predictedYCNN],[0]])

            #self.innervSignal = array([[predictedCNN[0,0]],[predictedCNN[0,1]],[0]])
            
            # Get the target rotation axis and angle from the model
            cameraRotAxis, cameraRotAngle = QuaiaOptican(self.eyeInitOrient, self.innervSignal, 0.001)
            totalCameraRotAxis = [cameraRotAxis[0] + self.initCameraRotAxis[0],
                                    cameraRotAxis[1] + self.initCameraRotAxis[1],
                                    cameraRotAxis[2] + self.initCameraRotAxis[2]]
            print "initialCameraRotAxis ", self.initCameraRotAxis
            print "cameraRotAxis ", cameraRotAxis
            print "initialCameraRotAngle ", self.initCameraRotAngle
            print "cameraRotationAngle ", cameraRotAngle*(180/pi)
            print "totalCameraRotAngle ", cameraRotAngle*(180/pi)+self.initCameraRotAngle
            print "totalCameraRotAxis ", totalCameraRotAxis

            '''cameraRotEuler = self.convertAxisAngleToEuler(cameraRotAxis,cameraRotAngle)
            initCameraRotEuler = self.convertAxisAngleToEuler(self.initCameraRotAxis,self.initCameraRotAngle)
            totalCameraRotEuler = initCameraRotEuler + cameraRotEuler
            print "initCameraRotEuler ", initCameraRotEuler
            print "cameraRotEuler ", cameraRotEuler
            print "two added together ", totalCameraRotEuler
            newCameraRotOrientation = np.zeros((3,1))


            angleThreshold = 0.191986
            if totalCameraRotEuler[0] < -angleThreshold or totalCameraRotEuler[0] > angleThreshold or totalCameraRotEuler[1] < -angleThreshold or totalCameraRotEuler[1] > angleThreshold:
                print "stuck? "
                if totalCameraRotEuler[0] < -angleThreshold:
                    newCameraRotOrientation[0] = -angleThreshold - initCameraRotEuler[0]
                if totalCameraRotEuler[0] > angleThreshold:
                    newCameraRotOrientation[0] = angleThreshold - initCameraRotEuler[0]
                if totalCameraRotEuler[1] < -angleThreshold:
                    newCameraRotOrientation[1] = -angleThreshold - initCameraRotEuler[1]
                if totalCameraRotEuler[1] > angleThreshold:
                    newCameraRotOrientation[1] = angleThreshold - initCameraRotEuler[1]
                [cameraRotAngle,cameraRotAxis[0],cameraRotAxis[1],_] = self.convertEulerToAxisAngle(newCameraRotOrientation[1],newCameraRotOrientation[2],newCameraRotOrientation[0])'''

            if cameraRotAngle*(180/pi) + self.initCameraRotAngle < -12:
                print "stuck?"
                cameraRotAngle = -0.20944 - (self.initCameraRotAngle*pi)/180
            if cameraRotAngle*(180/pi) + self.initCameraRotAngle > 12:
                print "stuck?"
                cameraRotAngle = 0.20944 - (self.initCameraRotAngle*pi)/180
            # update the eye's initial orientation for the next frame
            self.eyeInitOrient = self.convertAxisAngleToEuler(cameraRotAxis,cameraRotAngle)

            # convert rotation angle from radians to degrees for opengl rotation
            
            self.cameraRotAngle = cameraRotAngle*(180/pi)
            self.cameraRotAxis = cameraRotAxis

            # we are done dealing with the target 
            self.targetChanged = False

    # When the target has changed position, we need to rotate our eye accordingly
        if self.targetChanged:
            self.predictInnerv = True
            self.targetChanged = False


        glutSwapBuffers()

    def main(self):
        # initialization
        startTime = time()
        window = 0                                             # glut window number                            
        glutInit(sys.argv)                                             # initialize glut
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)                      # set window size
        glutInitWindowPosition(0, 0)                           # set window position
        window = glutCreateWindow("Our Eye")              # create window with title
        glutDisplayFunc(self.setUpSystem)                                  # set draw function callback
        glutIdleFunc(self.setUpSystem)                                     # draw all the time
        glutKeyboardFunc(self.keyPressed)
        glutMouseFunc(self.OnMouseClick)
        glutMainLoop()                                         # start everything

eyeCam = eyeCamera()
eyeCam.main()

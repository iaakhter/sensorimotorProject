from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import *
from math import *
from time import time

# References:
#         https://noobtuts.com/python/opengl-introduction
#         https://learnopengl.com/#!Getting-started/Camera
#         https://gist.github.com/strife25/803118

class eyeCamera:

	def __init__(self):
		self.width, self.height = 500, 500
		self.targetX, self.targetY, self.targetZ = 0, 0, 0
	
	def setUpCamera(self,cameraPosition,cameraTarget,cameraUp,
					perceivedTargetWidth,perceivedTargetHeight,
					cameraRotAngle,cameraRotAxis):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		
		glPushMatrix()
		# rotate the camera by the angle
		glRotate(cameraRotAngle,cameraRotAxis[0], cameraRotAxis[1], cameraRotAxis[2])
		gluLookAt(cameraPosition[0],cameraPosition[1],cameraPosition[2],
				  cameraTarget[0],cameraTarget[1],cameraTarget[2],
				  cameraUp[0],cameraUp[1],cameraUp[2])

		# let's look at the target that the camera perceives. This indicates
		# the region that is in focus by the camera
		self.perceivedTarget(perceivedTargetWidth,perceivedTargetHeight)
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
		color = (1.0,0.0,0.0)
		self.drawLine(0,0,0,0.5,0.0,0.0, color)
		color = (0.0,1.0,0.0)
		self.drawLine(0,0,0,0.0,0.5,0.0, color)
		color = (0.0,0.0,1.0)
		self.drawLine(0,0,0,0.0,0.0,0.5, color)

	def drawMovingTarget(self,width,height):
		startTime = time()%2.0
		xPosition = -1+startTime
		self.drawTarget(xPosition,0.0,0.0,width,height)


	# Draw a square representing the focus of the eye (along the optical axis)
	def perceivedTarget(self,width, height):
		color = (1.0,0.0,0.0)
		self.drawRectangle(0,0,0,width,height,color)

		# Draw the x,y and z axes
		color = (1.0,0.0,0.0)
		self.drawLine(0,0,0,0.5,0.0,0.0, color)
		color = (0.0,1.0,0.0)
		self.drawLine(0,0,0,0.0,0.5,0.0, color)
		color = (0.0,0.0,1.0)
		self.drawLine(0,0,0,0.0,0.0,0.5, color)
		

	def determineTargetOrientation(self,cameraPosition):
		#Get the screen coordinates for blue and red objects
		pixels = glReadPixels(0.0,0.0,self.width,self.height,format=GL_RGB,type=GL_FLOAT)
		#print "pixels"
		startBluex, startBluey = 0, 0
		endBluex, endBluey = 0, 0
		indicesB = where(pixels[:,:,2]==1.0)
		startBluey = min(indicesB[0])
		endBluey = max(indicesB[0])
		startBluex = min(indicesB[1])
		endBluex = max(indicesB[1])
		centerBluex = (startBluex + endBluex)/2.0
		centerBluey = (startBluey + endBluey)/2.0

		#Get the viewport, modelviewMatrix and the projectionMatrix
		viewport = glGetIntegerv (GL_VIEWPORT);						# //get actual viewport
	  	mvmatrix = glGetDoublev (GL_MODELVIEW_MATRIX);				# //get actual model view matrix
	  	projmatrix = glGetDoublev (GL_PROJECTION_MATRIX);			# //get actual projiection matrix

	  	#Convert the screen coordinates to word coordinates
	  	blueWorldC = gluUnProject(centerBluex,centerBluey,0.5,mvmatrix,projmatrix,viewport)
		targetOrientationVector = blueWorldC - cameraPosition
		targetAngleX = math.acos(dot(targetOrientationVector,array([1,0,0]))/(linalg.norm(targetOrientationVector)*linalg.norm(array([1,0,0]))))
		targetAngleY = math.acos(dot(targetOrientationVector,array([0,1,0]))/(linalg.norm(targetOrientationVector)*linalg.norm(array([0,1,0]))))
		targetAngleZ = math.acos(dot(targetOrientationVector,array([0,0,1]))/(linalg.norm(targetOrientationVector)*linalg.norm(array([0,0,1]))))
		targetOrientations = array([targetAngleX, targetAngleY, targetAngleZ])
		print "blue center screen: (",centerBluex,", ",centerBluey,")"
		print "blue center world: ",blueWorldC
		print "targetOrientationVector: ", targetOrientationVector
		print "targetOrientations(in radians): ", targetOrientations
		return targetOrientations


	def keyPressed(self,*args):
		if(args[0] == 'w'):
			self.targetY += 0.1
		elif(args[0] == 's'):
			self.targetY -= 0.1
		elif(args[0] == 'd'):
			self.targetX += 0.1
		elif(args[0] == 'a'):
			self.targetX -= 0.1
		


	#callback function for opengl
	def setUpSystem(self,):
		cameraPosition = array([0.0,0.0,1.0])
		cameraTarget = array([0.0,0.0,0.0])
		cameraUp = array([0.0,1.0,0.0])

		perceivedTargetWidth = 0.30
		perceivedTargetHeight = 0.30
		
		# just for the purposes of testing, moving angle
		cameraRotAngle = time()%(180)-90
		cameraRotAxis = array([1.0,0.0,0.0])

		# to make sure that we can actually see the whole target
		# at one point. Without this constraint, the transformation 
		# is too fast 
		if(cameraRotAngle >= -0.05 and cameraRotAngle <= 0.05):
			cameraRotAngle = 0.0
		# print "cameraRotationAngle ", cameraRotAngle

		cameraRotAngle = 127
		cameraRotAxis = array([7.07085373e-01, 7.07128178e-01, 1.21392305e-04])
		self.setUpCamera(cameraPosition,cameraTarget,cameraUp,
						perceivedTargetWidth,perceivedTargetHeight,
						cameraRotAngle,cameraRotAxis)
		
		targetWidth = 0.25
		targetHeight = 0.25
		glColor3f(0.0, 0.0, 1.0)
		#print "targetY: ", targetY
		self.drawTarget(self.targetX,self.targetY,self.targetZ,targetWidth,targetHeight)
		targetOrientation = self.determineTargetOrientation(cameraPosition)
		glutSwapBuffers()
		#print "done setting up camera"

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
		glutMainLoop()                                         # start everything

eyeCam = eyeCamera()
eyeCam.main()

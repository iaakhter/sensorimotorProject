from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import *
from math import *
from time import time
from quaiaoptican import *

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
		#color = (1.0,0.0,0.0)
		#self.drawLine(0,0,1,0,0,0,color)
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
		self.drawLine(x,y,z,x+0.5,y,z, color)
		color = (0.0,1.0,0.0)
		self.drawLine(x,y,z,x,y+0.5,z, color)
		color = (0.0,0.0,1.0)
		self.drawLine(x,y,z,x,y,z+0.5, color)

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

	def drawPositionOffocus(self, cameraPosition,perceivedTargetWidth,perceivedTargetHeight):
		mvmatrix = glGetDoublev (GL_MODELVIEW_MATRIX);
		invmvmatrix = linalg.inv(mvmatrix)
		visualAxis = array([mvmatrix[2][0], mvmatrix[2][1], mvmatrix[2][2]])
		pointOffocus = cameraPosition - visualAxis
		color = (1.0,0.0,0.0)
		self.drawRectangle(pointOffocus[0],pointOffocus[1],pointOffocus[2],perceivedTargetWidth,perceivedTargetHeight,color)
		

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
	  	projmatrix = glGetDoublev (GL_PROJECTION_MATRIX);			# //get actual projection matrix

	  	#Convert the screen coordinates to world coordinates
	  	blueWorldC = gluUnProject(centerBluex,centerBluey,0.5,mvmatrix,projmatrix,viewport)
		targetVector = around(blueWorldC, decimals = 2) - cameraPosition
		# find the amount that the eye would need to rotate to look at the
		# target assuming its initial orientation is [0,0,0]
		targetVector_magnitude = linalg.norm(targetVector)
		angleX = atan(targetVector[1]/sqrt(targetVector_magnitude**2 - targetVector[1]**2))
		angleY = atan(targetVector[0]/targetVector[2])
		targetOrientations = array([angleX, angleY, 0.0])
		return targetOrientations

	def determineRequiredInnerv(self, targetOrientation):
		print "targetOrient: ", targetOrientation
		print "eyeInitOrient: ", self.eyeInitOrient
		innervationSignal= array([(targetOrientation[0] - self.eyeInitOrient[0])*1000,
						   (targetOrientation[1] - self.eyeInitOrient[1])*1000,
						   (targetOrientation[2] - self.eyeInitOrient[2])*1000])
		# because the model cannot handle a zero array for innervation signal
		if(sum(innervationSignal) == 0):
			diffOrientation[0] = 0.00000001
		print "innervationSignal: ", innervationSignal
		return innervationSignal


	def keyPressed(self,*args):
		if(args[0] == 'w'):
			self.targetChanged = True
			self.targetY += 0.1
		elif(args[0] == 's'):
			self.targetChanged = True
			self.targetY -= 0.1
		elif(args[0] == 'd'):
			self.targetChanged = True
			self.targetX += 0.1
		elif(args[0] == 'a'):
			self.targetChanged = True
			self.targetX -= 0.1


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
		
		# When the target has changed position, we need to rotate our eye accordingly
		if(self.targetChanged):
			# Get the orientation that the eye needs to be at to see the target
			targetOrientation = self.determineTargetOrientation(cameraPosition)
			
			# Convert the target orientation to the required innervation signal
			# This is what we need to learn (Righ now I just have a simple linear mapping
			# from difference between target and initial orientation to the innervation
			# signal)
			self.innervSignal = self.determineRequiredInnerv(targetOrientation)
			
			# Get the target rotation axis and angle from the model
			cameraRotAxis, cameraRotAngle = QuaiaOptican(self.eyeInitOrient, self.innervSignal, 0.001)
			
			# update the eye's initial orientation for the next frame
			self.eyeInitOrient = self.convertAxisAngleToEuler(cameraRotAxis,cameraRotAngle)

			# convert rotation angle from radians to degrees for opengl rotation
			cameraRotAngle = cameraRotAngle*(180/pi)
		
			self.cameraRotAngle = cameraRotAngle
			self.cameraRotAxis = cameraRotAxis

			# we are done dealing with the target 
			self.targetChanged = False
			print "self.initCameraRotAngle:", self.initCameraRotAngle
			print "self.initCameraRotAxis:", self.initCameraRotAxis
			print "self.cameraRotAngle:", self.cameraRotAngle
			print "self.cameraRotAxis:", self.cameraRotAxis
			#print "innervationSignal: ", self.innervSignal
			#print "correspoinding eyeOrientation: ", self.eyeInitOrient
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
		glutMainLoop()                                         # start everything

eyeCam = eyeCamera()
eyeCam.main()

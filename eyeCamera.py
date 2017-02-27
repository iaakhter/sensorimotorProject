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
		self.cameraRotAxis = array([0,0,0])
		self.cameraRotAngle = 0
	
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
		targetVector = around(blueWorldC, decimals = 2) - cameraPosition
		targetOrientationVector = targetVector - array([0,0,-1])
		targetAngleX = math.acos(dot(targetOrientationVector,array([1,0,0]))/(linalg.norm(targetOrientationVector)*linalg.norm(array([1,0,0]))))
		if isnan(targetAngleX):
	 		targetAngleX = 0.000000001
		targetAngleY = math.acos(dot(targetOrientationVector,array([0,1,0]))/(linalg.norm(targetOrientationVector)*linalg.norm(array([0,1,0]))))
		if isnan(targetAngleY):
	 		targetAngleY = 0.000000001
		targetAngleZ = math.acos(dot(targetOrientationVector,array([0,0,1]))/(linalg.norm(targetOrientationVector)*linalg.norm(array([0,0,1]))))
		if isnan(targetAngleZ):
	 		targetAngleZ = 0.000000001
		targetOrientations = array([targetAngleX, targetAngleY, targetAngleZ])
		print "blue center screen: (",centerBluex,", ",centerBluey,")"
		print "blue center world: ",blueWorldC
		print "targetOrientationVector: ", targetOrientationVector
		print "targetOrientations(in radians): ", targetOrientations
		return targetOrientations

	def determineRequiredInnerv(self, targetOrientation):
		# because the model cannot handle a zero array for innervation signal
		if(sum(targetOrientation) == 0):
			targetOrientation[0] = 0.00000001
		return array([[targetOrientation[0]],[targetOrientation[1]],[targetOrientation[2]]])


	def keyPressed(self,*args):
		if(args[0] == 'w'):
			self.targetY += 0.1
		elif(args[0] == 's'):
			self.targetY -= 0.1
		elif(args[0] == 'd'):
			self.targetX += 0.1
		elif(args[0] == 'a'):
			self.targetX -= 0.1


	# taken from http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToEuler/
	def convertAxisAngleToEuler(self,rotAxis, rotAngle):
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

		# convert angle from radians to degrees for opengl rotation
		self.cameraRotAngle = self.cameraRotAngle*(180/pi)
		print "cameraRotAngle ", self.cameraRotAngle
		self.setUpCamera(cameraPosition,cameraTarget,cameraUp,
						perceivedTargetWidth,perceivedTargetHeight,
						self.cameraRotAngle,self.cameraRotAxis)
		
		targetWidth = 0.25
		targetHeight = 0.25
		self.drawTarget(self.targetX,self.targetY,self.targetZ,targetWidth,targetHeight)

		targetOrientation = self.determineTargetOrientation(cameraPosition)
		self.innervSignal = self.determineRequiredInnerv(targetOrientation)
		self.cameraRotAxis, self.cameraRotAngle = QuaiaOptican(self.eyeInitOrient, self.innervSignal, 0.001)
		self.eyeInitOrient = self.convertAxisAngleToEuler(self.cameraRotAxis, self.cameraRotAngle)

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

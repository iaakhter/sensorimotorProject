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
width, height = 500, 500
def setUpCamera(cameraPosition,cameraTarget,cameraUp):
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	#gluPerspective(400.,1.,1.,400.)
	#set up view matrix
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	#eye position (0.0,0.0,3.0)
	#reference point (0.0,0.0,0.0)
	#up vector(0.0,1.0,0.0)
	angle = time()%(math.pi)-math.pi/2.0

	# to make sure that we can actually see the whole target
	# at one point. Without this constraint, the transformation 
	# is too fast and we can hardly see the whole object when
	# we are supposed to see it
	if(angle >= -0.05 and angle <= 0.05):
		angle = 0.0
	#print "angle ", angle
	glRotate(angle,0.0, 1.0, 0.0)
	gluLookAt(cameraPosition[0],cameraPosition[1],cameraPosition[2],
			  cameraTarget[0],cameraTarget[1],cameraTarget[2],
			  cameraUp[0],cameraUp[1],cameraUp[2])

def drawRectangle(x, y, z, width, height,color):
	glColor3f(color[0],color[1],color[2])
	glBegin(GL_QUADS)
	glVertex3f(x-width/2.0, y-height/2.0, z)                               
	glVertex3f(x+width/2.0, y-height/2.0, z)                      
	glVertex3f(x+width/2.0, y+height/2.0, z)            
	glVertex3f(x-width/2.0, y+height/2.0, z)                 
	glEnd()

def drawTarget(x,y,z,width,height):
	color = (0.0,0.0,1.0)
	drawRectangle(x,y,z,width,height,color)

def drawMovingTarget(width,height):
	startTime = time()%2.0
	xPosition = -1+startTime
	drawTarget(xPosition,0.0,0.0,width,height)


#Draw a square representing the focus of the eye (along the optical axis)
def perceivedTarget(width, height):
	color = (1.0,0.0,0.0)
	drawRectangle(0,0,0.0,width,height,color)

def determineDirection(targetWidth, targetHeight, eyeCenterWidth, eyeCenterHeight):
	# Expected dimensions
	targetPixelWidth = width*targetWidth
	targetPixelHeight = height*targetHeight
	eyeCenterPixelWidth = width*eyeCenterWidth
	eyeCenterPixelHeight = height*eyeCenterHeight
	pixels = glReadPixels(0.0,0.0,width,height,format=GL_RGB,type=GL_FLOAT)
	print "pixels"
	startBluex, startBluey = 0, 0
	endBluex, endBluey = 0, 0
	startRedx, startRedy = 0, 0
	endRedx, endRedy = 0, 0
	indicesB = where(pixels[:,:,2]==1.0)
	indicesR = where(pixels[:,:,0]==1.0)
	startBluex = min(indicesB[0])
	endBluex = max(indicesB[0])
	startBluey = min(indicesB[1])
	endBluey = max(indicesB[1])
	startRedx = min(indicesR[0])
	endRedx = max(indicesR[0])
	startRedy = min(indicesR[1])
	endRedy = max(indicesR[1])
	#print "target positions: (",startBluex,", ",startBluey,"), (",endBluex,", ",endBluey,")"
	#print "eyecenter positions: (",startRedx,", ",startRedy,"), (",endRedx,", ",endRedy,")"
	#print "expected targetPixel - width: ", targetPixelWidth, " height: ",targetPixelHeight
	#print "expected eyeCenterPixel - width: ", eyeCenterPixelWidth, " height: ",eyeCenterPixelHeight
	#print "rectangleWidth: ", endBluey - startBluey

	'''For horizontal direction '''
	if(abs(endBluey - startRedy) > abs(endRedy - startBluey)):
		print "Need to move right"
	elif(abs(endBluey - startRedy) < abs(endRedy - startBluey)):
		print "Need to move left"
	else:
		print "Eye is in correct position"

#callback function for opengl
#From the camera's point of view the object should always be at (0,0,0)
# so we will say that it is in focus
def setUpSystem():
	cameraPosition = array([0.0,0.0,1.0])
	cameraTarget = array([0.0,0.0,0.0])
	cameraUp = array([0.0,1.0,0.0])
	setUpCamera(cameraPosition,cameraTarget,cameraUp)
	
	targetWidth = 0.5
	targetHeight = 0.25
	glColor3f(0.0, 0.0, 1.0)
	drawTarget(0.0,0.0,0.0,targetWidth,targetHeight)
	#Move the target horizontally
	#drawMovingTarget(width,height)
	eyeCenterWidth = 0.02
	eyeCenterHeight = 0.02
	perceivedTarget(eyeCenterWidth,eyeCenterHeight)
	determineDirection(targetWidth,targetHeight,eyeCenterWidth,eyeCenterHeight)
	glutSwapBuffers()
	#print "done setting up camera"

def main():
	# initialization
	startTime = time()
	window = 0                                             # glut window number                            
	glutInit()                                             # initialize glut
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
	glutInitWindowSize(width, height)                      # set window size
	glutInitWindowPosition(0, 0)                           # set window position
	window = glutCreateWindow("Our Eye")              # create window with title
	glutDisplayFunc(setUpSystem)                                  # set draw function callback
	glutIdleFunc(setUpSystem)                                     # draw all the time
	glutMainLoop()                                         # start everything

main()

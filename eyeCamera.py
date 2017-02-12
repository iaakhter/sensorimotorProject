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
	gluLookAt(cameraPosition[0],cameraPosition[1],cameraPosition[2],
			  cameraTarget[0],cameraTarget[1],cameraTarget[2],
			  cameraUp[0],cameraUp[1],cameraUp[2])

def drawTarget(x, y, z, width, height):
	glColor3f(0.0, 0.0, 1.0)
	glBegin(GL_QUADS)
	glVertex3f(x-width/2.0, y-height/2.0, z)                               
	glVertex3f(x+width/2.0, y-height/2.0, z)                      
	glVertex3f(x+width/2.0, y+height/2.0, z)            
	glVertex3f(x-width/2.0, y+height/2.0, z)                 
	glEnd()

def drawFilledCircle(x, y, radius):
	triangleAmount = 20 # of triangles used to draw circle
	twicePi = 2.0 * pi
	
	glBegin(GL_TRIANGLE_FAN)
	glVertex3f(x, y,0.0) # center of circle
	for i in range(triangleAmount):
		glVertex3f(x + (radius * cos(i *  twicePi / triangleAmount)), y + (radius * sin(i * twicePi / triangleAmount)),0.0)
	glEnd()

#Draw a circle representing the focus of the eye (along the optical axis)
def perceivedTarget():
	glColor3f(1.0, 0.0, 0.0)
	drawFilledCircle(0,0,0.01)

#callback function for opengl
def setUpSystem():
	startTime = time()%2.0
	#radius = 0.1
	#camX = sin(startTime)*radius
	#camZ = cos(startTime)*radius
	cameraPosition = array([0.0,0.0,1.0])
	cameraTarget = array([0.0,0.0,0.0])
	cameraUp = array([0.0,1.0,0.0])
	#cameraPosition[0] = camX
	#cameraPosition[2] = camZ
	setUpCamera(cameraPosition,cameraTarget,cameraUp)
	
	#Move the target horizontally
	xPosition = -1+startTime
	drawTarget(xPosition,0.0,0.0,0.5,0.25)
	perceivedTarget()
	glutSwapBuffers()
	print "done setting up camera"

def main():
	# initialization
	width, height = 500, 400
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

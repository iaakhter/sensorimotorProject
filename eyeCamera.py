from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import *
from sklearn.preprocessing import normalize

# References:
#         https://noobtuts.com/python/opengl-introduction
#         https://learnopengl.com/#!Getting-started/Camera

width, height = 500, 400   

def setUpCamera(cameraPosition,cameraTarget,cameraUp):
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(40.,1.,1.,40.)
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
	glBegin(GL_QUADS)
	glVertex3f(x, y, z)                               
	glVertex3f(x+width, y, z)                      
	glVertex3f(x+width, y+height, z)            
	glVertex3f(x, y+height, z)                   
	glEnd()
	

def setUpSystem():
	cameraPosition = array([0.0,0.0,3.0])
	cameraTarget = array([0.0,0.0,0.0])
	cameraUp = array([0.0,1.0,0.0])
	setUpCamera(cameraPosition,cameraTarget,cameraUp)
	glColor3f(0.0, 0.0, 1.0)
	drawTarget(0.0,0.0,0.0,200,100)
	glutSwapBuffers()
	print "done setting up camera"

def main():
	# initialization
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

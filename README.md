@authors: 
Ariadna Estrada and Itrat Akhter

This is a python program that simulates eye saccades using supervised learning (artificial neural network) in an OpenGL environment. 
You need to have PyOpenGL, numpy, sci-kit learn, opencv, tensorflow and keras installed to run this program.
Run eyeCameraMl.py for simulation. Blue box represents the target. The red box represents the gaze of the camera which represents the eye in our simulation. The user can move the target using mouse clicks. When the target is moved, appropriate innervation signals for an eye model are predicted. The eye model outputs a saccade in the form of a rotation with the predicted innervation signals as inputs. The camera rotates/performes the saccade towards the moved target. This causes the red box representing the gaze of the camera to move. When the red box, surrounds the target, the camera is seeing the target perfectly. Otherwise, there is an error in the saccade. 

In the default mode, the user is only able to see the predictions by the ANN. If the user wants to view the prediction by the CNN, they can press ``n" on the keyboard. To go back to the ANN, the user can press ``c".

More details about the neural networks and our methods are included in the final report in this repo. The user can also see a video of our program which is also included in this repo. 
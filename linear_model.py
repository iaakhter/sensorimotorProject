#modified by Itrat Ahmed Akhter
#CPSC 340 hw 3
#linear_model.py


import numpy as np
from numpy.linalg import solve
import sys
from scipy.optimize import approx_fprime
import math


# Original Least Squares
class LeastSquares:
    # Class constructor
    def __init__(self):
        pass

    def fit(self,X,y):
        # Solve least squares problem

        a = np.dot(X.T, X)
        b = np.dot(X.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):

        w = self.w
        yhat = np.dot(Xhat, w)
        return yhat

# Least Squares with a bias added
class LeastSquaresBias:
    def __init__(self):
        pass

    def fit(self,X,y):

        ''' YOUR CODE HERE FOR Q2.1 '''
        # add a column of one to X
        Z = np.ones((X.shape[0],X.shape[1]+1))
        Z[:,1:Z.shape[1]] = X
        # Solve least squares problem
        a = np.dot(Z.T, Z)
        b = np.dot(Z.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):

        ''' YOUR CODE HERE FOR Q2.1 '''
        w = self.w
        Zhat = np.ones((Xhat.shape[0],Xhat.shape[1]+1))
        Zhat[:,1:Zhat.shape[1]] = Xhat
        yhat = np.dot(Zhat, w)
        return yhat

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):

        ''' YOUR CODE HERE FOR Q2.2 '''
        Z = self.__polyBasis(X)
        a = np.dot(Z.T, Z)
        b = np.dot(Z.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):

        ''' YOUR CODE HERE FOR Q2.2 '''
        w = self.w
        Zhat = self.__polyBasis(Xhat)
        yhat = np.dot(Zhat, w)
        return yhat

    # A private helper function to transform any matrix X into 
    # the polynomial basis defined by this class at initialization 
    # Returns the matrix Z that is the polynomial basis of X.   
    def __polyBasis(self, X):

        n = X.shape[0]
        d = self.p + 1
        # Z should have as many rows as X and as many columns as (p+1)
        Z = np.ones((n, d))

        ''' YOUR CODE HERE FOR Q2.2'''
        # Fill in Z
        X = np.reshape(X,X.shape[0])
        for i in range(1,d):
            Z[:,i] = np.power(X,i)

        return Z

# Least Squares with RBF Kernel
class LeastSquaresRBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self,X,y):
        self.X = X
        [n, d] = X.shape

        Z = self.__rbfBasis(X, X, self.sigma)

        # Solve least squares problem
        l = 1e-12

        a = Z.T.dot(Z) + l* np.identity(n)
        b = np.dot(Z.T, y)
        self.w = solve(a,b)

    def predict(self, Xtest):
        Z = self.__rbfBasis(Xtest, self.X, self.sigma)
        yhat = Z.dot(self.w)
        return yhat

    def __rbfBasis(self, X1, X2, sigma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        den = 1 / np.sqrt(2 * np.pi * (sigma** 2))

        D = (X1**2).dot(np.ones((d, n2))) + \
            (np.ones((n1, d)).dot((X2.T)** 2)) - \
            2 * (X1.dot( X2.T))

        Z = den * np.exp(-1* D / (2 * (sigma**2)))
        return Z

# Least Squares where each sample point X has a weight associated with it.
class WeightedLeastSquares:

    def __init__(self):
        pass

    def fit(self,X,y,z):

        ''' YOUR CODE HERE FOR Q4.1 '''
        a = np.dot(X.T, z)
        a = np.dot(a,X)
        b = np.dot(X.T, z)
        b = np.dot(b,y)
        self.w = solve(a, b)

    def predict(self,Xhat):
        '''YOUR CODE HERE FOR Q4.1 '''
        w = self.w
        yhat = np.dot(Xhat, w)
        return yhat


class LinearModelGradient:

    def __init__(self):
        pass

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')
        

        self.w, f = findMin.findMin(self.funObj, self.w, 100, X, y)

    def predict(self,Xtest):

        w = self.w
        yhat = Xtest*w
        return yhat

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE FOR Q4.3 '''

        # Calculate the function value
        '''f = (1/2)* np.sum((X.dot(w)-y)**2)
        f = np.sum(math.exp(X.dot(w)-y) + math.exp(y-X.dot()))

        # Calculate the gradient value
        g = X.T.dot(X.dot(w) - y)'''

         # Calculate the function value
        helperFunc = X.dot(w)-y
        f = 0
        for i in range(X.shape[0]):
            f += math.log(math.exp(helperFunc[i]) + math.exp(-helperFunc[i]))

        # Calculate the gradient value
        g = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            g += ((math.exp(helperFunc[i]) - math.exp(-helperFunc[i]))/(math.exp(helperFunc[i]) + math.exp(-helperFunc[i])))*X[i]

        return (f,g)
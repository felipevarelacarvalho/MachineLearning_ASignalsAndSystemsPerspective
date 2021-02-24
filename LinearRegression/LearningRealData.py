#import libraries
import numpy as np
import matplotlib.pyplot as plt 
import timeit
import sys
from SimulatedSyntheticData import SimulateSyntheticData
from SimulatedSyntheticData import SimulateSytheticNonZeroMean
from SimulatedSyntheticData import SimulateSytheticDiagonalCovariance
from TrainingMethods import PseudoInverse
from TrainingMethods import NormalEquation
from TrainingMethods import GradientDescent

data = np.loadtxt(r'C:\Projects\EE425\MachineLearning_ASignalsAndSystemsPerspective\LinearRegression\airfoil_self_noise.dat')

y = [row[5] for row in data]

x = np.delete(data, -1, axis = 1)

np.set_printoptions(threshold = sys.maxsize)
theta_array  = np.array([[1], [1], [1], [1], [1]]) #Not necessary for this part, used here for debuging purposes
y = np.array(y)
y = y.reshape(-1,1)

start = timeit.default_timer() #Starts  timer
theta = GradientDescent(x, y, theta_array , 10**-3, 20)
stop = timeit.default_timer() #Stops timer
print('Timer for learning theta is seconds: ', stop - start) #prints time

#Calculate error
error = 0;
theta_approx  = theta[0]
print(theta_approx)
for i in range(len(y)):
    error = error +( (y[i] - np.matmul(theta_approx.transpose(), x[i]))**2 / len(y) )
print(error)
#import libraries
import numpy as np
import matplotlib.pyplot as plt 
import timeit
from SimulatedSyntheticData import SimulateSyntheticData
from SimulatedSyntheticData import SimulateSytheticNonZeroMean
from SimulatedSyntheticData import SimulateSytheticDiagonalCovariance
from TrainingMethods import PseudoInverse
from TrainingMethods import NormalEquation
from TrainingMethods import GradientDescent

data = np.loadtxt(r'C:\Projects\EE425\MachineLearning_ASignalsAndSystemsPerspective\LinearRegression\airfoil_self_noise.dat')
print(len(data))

y = [row[5] for row in data]
x = np.delete(data, 1, 1)
#theta = GradientDescent(x, y,  np.array([[1], [4], [2], [10], [23]]), 0.001, 1000)
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

#Parameters that we will be using for data generation
#Our data will have x ∼ N (0, I), therefore, mean zero and covariance identity
m = 80
n = 100
m_test = 100
theta = np.zeros(n)
mu = np.zeros(n)

#θ = [100, −99, 98, −97...1]'
for i in range(n):
    if (i%2 == 0):
        theta[i] = 100 - i
    else:
        theta[i] = - 100 + i
theta = theta.transpose()

sigma_epsilon_squared  = 0.01*( (np.linalg.norm(theta))**2 )

#Generating training data
training_data = SimulateSyntheticData(m, n, mu, sigma_epsilon_squared, theta)

#Learn theta
theta_hat = PseudoInverse(training_data[0], training_data[2], theta)

#Generate testing data
testing_data = SimulateSyntheticData(m_test, n, mu, sigma_epsilon_squared, theta)

#Predict y_hat
y_hat_test = np.matmul(theta_hat[0].transpose(), testing_data[0]) #y_hat = (theta_hat')(x_test)


y_test = testing_data[2]
x_test = testing_data[0]

#Monte Carlo
#Expected value (ytest - y_hattest)**2
monte_carlo_numerator = 0
monte_carlo_denominator = 0
for i in range(m_test):
    monte_carlo_numerator = monte_carlo_numerator + (y_test[i] - y_hat_test[i])**2
monte_carlo_numerator = monte_carlo_numerator/ m_test
#Expected value (ytest)**2
for i in range(m_test):
    monte_carlo_denominator = monte_carlo_denominator + (y_test[i]**2)
monte_carlo_denominator = monte_carlo_denominator/ m_test

monte_carlo_error = monte_carlo_numerator/monte_carlo_denominator
print(monte_carlo_error)

estimation_error  = ( np.linalg.norm(theta - theta_hat[0]) )**2 / ( (np.linalg.norm(theta))**2 )
print(estimation_error)


plot = plt.gca();
plot.scatter(80, monte_carlo_error, label = "part_a m = 30, Pseudo-Inverse");
for index, value in enumerate([1]):
    plot.annotate(monte_carlo_error, (80, monte_carlo_error))
# naming the x axis 
plt.xlabel('m values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Part 1 results for a') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 
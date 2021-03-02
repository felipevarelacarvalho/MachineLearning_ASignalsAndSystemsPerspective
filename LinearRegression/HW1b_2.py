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
from MonteCarloEstimate import MonteCarloEstimate

#Parameters that we will be using for data generation
#Our data will have x ∼ N (0, I), therefore, mean zero and covariance identity
m = 80
n = 100
m_test = round(0.2*m)
m_total = m - m_test


#Start plot
plot = plt.gca();

#Set up training data and training data:
#--------------------------------------------------------------------------------------------
#θ = [100, −99, 98, −97...1]'
theta = np.zeros(n)
for i in range(n):
    if (i%2 == 0):
        theta[i] = 100 - i
    else:
        theta[i] = - 100 + i
theta = theta.transpose()

mu = np.zeros(n)
sigma_epsilon_squared  = 0.1*( (np.linalg.norm(theta))**2 )

#Generating training data
training_data = SimulateSyntheticData(m, n, mu, sigma_epsilon_squared, theta)

#Generate testing data
testing_data = SimulateSyntheticData(m_test, n, mu, sigma_epsilon_squared, theta)
#--------------------------------------------------------------------------------------------


for n_small in range(2, n):
    x_vector = training_data[0]
    y_vector = training_data[2]
    x_test_vector = testing_data[0]
    y_test_vector = testing_data[2]

    #Create new vectors deleting n - n_small + 1 data values
    theta_new = np.zeros(n_small)
    x_new = np.zeros((m,n_small))
    x_test_new = np.zeros((m_test, n_small))
   

    #Fill in new matricies with deleted values
    i = 0
    for i in range(m):
        for j in range(n_small):
            x_new[i][j] = x_vector[i][j]
            

    i = 0
    for i in range(m_test):
        for j in range(n_small):
            x_test_new[i][j] = x_test_vector[i][j]

    i = 0 
    for i in range(n_small):
        theta_new[i] = theta[i]

    
    #Learn theta
    theta_hat = GradientDescent(x_new, y_vector, theta_new, 0.001, 1000)

    #Predict y_hat from testing data
    y_hat_test = np.matmul(theta_hat[0], x_test_new.transpose()).transpose() #y_hat = (theta_hat')(x_test)

    #Monte Carlo
    y_test = y_test_vector
    x_test = x_test_new
    
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

    #errors_m80  = MonteCarloEstimate(m_total, n_small , m_test, mu, theta, sigma_epsilon_squared)
    #if(errors_m80[0] < 10):
    plot.bar(n_small, monte_carlo_error, color=(0.2, 0.4, 0.6, 0.6));
    print("Monte Carlo approximation when m = 80 and n_small = ", n_small, ": ", monte_carlo_error)

# naming the x axis 
plt.xlabel('n_small values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Monte Carlo error, sigma squared = 0.1. Training method, Gradient Descent ') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.yscale("log")
plt.show() 

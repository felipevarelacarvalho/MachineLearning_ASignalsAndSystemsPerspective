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

def MonteCarloEstimate(m, n, m_test, mu, theta, sigma_epsilon_squared):
    #Parameters that we will be using for data generation
    #Our data will have x ∼ N (0, I), therefore, mean zero and covariance identity

    #Generating training data
    training_data = SimulateSyntheticData(m, n, mu, sigma_epsilon_squared, theta)

    #Learn theta
    theta_hat = NormalEquation(training_data[0], training_data[2], theta)

    #Generate testing data
    testing_data = SimulateSyntheticData(m_test, n, mu, sigma_epsilon_squared, theta)
    
    #Predict y_hat from testing data
    y_hat_test = np.matmul(theta_hat[0], testing_data[0].transpose()).transpose() #y_hat = (theta_hat')(x_test)

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
    
    #Estimation error using predicted theta
    estimation_error  = ( np.linalg.norm(theta - theta_hat[0], 2) )**2 / ( (np.linalg.norm(theta))**2 )

    return monte_carlo_error , estimation_error, theta_hat
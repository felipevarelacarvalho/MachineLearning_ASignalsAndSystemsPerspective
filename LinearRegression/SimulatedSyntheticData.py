#import libraries
import numpy as np

def SimulateSyntheticData(m, n, mu, covariance_squared, theta):
    x_array  = np.zeros((m,n))   
    error_array = np.zeros((m,1))
    y_array = np.zeros((m,n))
    i = 0;
    j = 0;

    #Generate data
    for i in range(m):
        for j in range(n):
            x_array[i][j] = np.random.normal(mu, len(mu), 1) 
        error_array[i] = np.random.normal(0, covariance_squared, 1)
    y_array = np.dot(x_array, theta) + error_array

    #Test data generate
    x_array_true  = np.zeros((m,n))   
    error_array_true = np.zeros((m,1))
    y_array_true = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            x_array_true[i][j] = np.random.normal(mu, len(mu), 1) 
        error_array_true[i] = np.random.normal(0, covariance_squared, 1)
    y_array_true = np.dot(x_array_true, theta) + error_array_true
    
    return x_array, error_array, y_array, x_array_true, error_array_true, y_array_true
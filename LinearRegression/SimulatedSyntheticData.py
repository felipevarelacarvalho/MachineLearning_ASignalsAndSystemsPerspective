#import libraries
import numpy as np

def SimulateSyntheticData(m, n, mu, covariance_squared, theta):
    x_array  = np.zeros((m,n))   
    error_array = np.zeros((m,1))
    y_array = np.zeros(m)
    i = 0;
    j = 0;

    mu_mean = np.mean(mu)
    
    #Generate data
    for i in range(m):
        for j in range(n):
            x_array[i][j] = np.random.randn(1, 1) + mu[j] 
        error_array[i] = np.random.randn(1,1) * ( covariance_squared**(1/2) )
        y_array[i] = np.matmul(theta.transpose(), x_array[i]) + error_array[i]

    #Test data generate
    x_array_true  = np.zeros((m,n))   
    error_array_true = np.zeros((m,1))
    y_array_true = np.zeros(m)
    for i in range(m):
        for j in range(n):
            x_array_true[i][j] = np.random.randn(1, 1) + mu[j] 
        error_array_true[i] = np.random.randn(1,1) * ( covariance_squared**(1/2) )
    y_array_true = np.matmul(theta.transpose(), x_array_true[i]) + error_array_true[i]
    
    return x_array, error_array, y_array, x_array_true, error_array_true, y_array_true

def SimulateSytheticNonZeroMean(m, n, mu, covariance_squared, theta):

    theta_tilde = np.zeros((len(theta)+1, 1)) 
    theta_tilde[0] = np.mean(theta)
    for i in range(len(theta)):
        theta_tilde[i+1] = theta[i]
    theta_tilde = theta_tilde.reshape(-1,1)

    mu_mean = np.mean(mu)

    x_array  = np.zeros((m,n+1))   
    error_array = np.zeros((m,1))
    y_array = np.zeros((m,1))

    #Create x tilde
    for i in range(m):
        x_array[i][0] = 1
    #Generate data
    for i in range(m):
        for j in range(n):
            x_array[i][j + 1] = np.random.normal(mu_mean, 1, 1) 
        error_array[i] = np.random.normal(0, covariance_squared, 1)
    y_array = np.dot(x_array, theta_tilde) + error_array

    return x_array, error_array, y_array, theta_tilde

def SimulateSytheticDiagonalCovariance(m, n, mu, covariance_matrix, sigma_error_squared, theta):
    x_array = np.random.multivariate_normal(mu, covariance_matrix, size = (m,n))
    error_array = np.random.normal(0, sigma_error_squared, size = (m,1))
    y_array = np.dot(x_array, theta)

    return x_array, error_array, y_array
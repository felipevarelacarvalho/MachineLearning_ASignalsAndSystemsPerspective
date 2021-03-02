#Code heavily inspired by Praneeth's code
#import libraries
import numpy as np

def generate_data_classification(m, n , phi):
    #Inputs
        #m = number of samples, n = number of features, phi: fraction of samples that are from class 0"

    #Outputs
        #X: matrix of features (mxn), Y: class labels (m x 1) where each eantry is either 0 or 1

    #Generate mean vectors and covariance matrix
    mu_0 = 1 * np.ones((n,1)) #separation between means, data will be considered seperable
    mu_1 = -1 * np.ones((n,1))
    sigma = 3 * np.eye(n)

    #Generate class lables
    rand_data = np.random.rand(m, 1) #Generates samples from uniform random distribution (between 0, 1)
    y = 1 * (rand_data >= phi) #vector of size m, each entry is 0,1

    #Generate X
    X = np.zeros((m, n))
    for i in range (m):
        if(y[i] == 0):
            x_val = sigma @ np.random.randn(n, 1) + mu_0
        else:
            x_val = sigma @ np.random.randn(n, 1) + mu_1
        X[i:i + 1, :] = x_val.T
    
    #print (X)
    #print(y)
    return X, y
    
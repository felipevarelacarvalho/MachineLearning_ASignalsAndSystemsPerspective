#import libraries
import numpy as np

def PseudoInverse(X, y, theta_true):
    #pseudo-inverse
    theta_hat_pinv = np.dot(np.linalg.pinv(X), y) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
    return theta_hat_pinv, (np.linalg.norm(theta_hat_pinv -theta_true))

def NormalEquation(X, y, theta_true):
    #normal equation
    theta_hat_norm = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)),y)
    return theta_hat_norm, (np.linalg.norm(theta_hat_norm - theta_true))


def calculate_cost_gradient (X, y, theta):
    y_hat = np.squeeze(X).dot(np.squeeze(theta))

    cost = 2 * (y_hat - y) * x.T #\Del_(\theta) J(\theta) = (theta^T x - y)x^T
    return cost

def cost_function(x, y, theta):
    m = len(y)
    y_hat = np.dot(x, theta)
    cost = (1/2*m)*np.sum(np.square(y_hat - y))
    return cost

def GradientDescent(X, y, theta_true, step_size, max_iter):
    #Input
        #X = Matrix of X, y = vector of y, theta_true = vector with weight theta, learning_rate, max_iter
    #Output
        #Theta hat = learned theta, #Normalized error

    
    theta_hat_gd = theta_true
    m = len(y)
    cost_error = np.zeros(max_iter)

    for i in range(max_iter):
        theta_approx = np.dot(X, theta_true)
        theta_hat_gd = theta_hat_gd - (1/m) * step_size *(X.transpose().dot((theta_approx - y)))
        cost_error[i] = cost_function(X, y,theta_hat_gd)
        

    return theta_hat_gd, (np.linalg.norm(theta_hat_gd - theta_true)), cost_error
    #return theta
    
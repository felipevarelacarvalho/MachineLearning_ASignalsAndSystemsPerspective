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

def GradientDescent(X, y, theta_true, mu, max_iterations):
    #gradient descent
    theta_hat_gd = np.zeros_like(theta_true)
    for t in range(max_iter):
        theta_hat_gd = theta_hat_gd - mu *(np.dot(np.dot(np.transpose(X), X),theta_hat_gd) - np.dot(np.transpose(X),y))
    return theta_hat_gd, (np.linalg.norm(theta_hat_gd - theta_true))
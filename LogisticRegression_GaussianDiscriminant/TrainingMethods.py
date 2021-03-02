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
    y_hat = np.dot(X, theta)
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
    
def sigmoid(z):
    return (1/(1 + np.exp(-z)))

def neg_log_cost(X, y, theta):
    #Input
        #X = feature matrix, y = class label, theta = hyperplane
    
    cost_eval = 0
    m, n = X.shape
    for i in range(m):
        xtmp = X[i:i + 1, :]
        htheta_xtmp = sigmoid(np.squeeze(xtmp).dot(np.squeeze(theta)))  #h_{theta}(x)
        cost_eval -= y[i] + np.log(htheta_xtmp + np.finfo(float).eps) + (1 - y[i]) * np.log(1 - htheta_xtmp + np.finfo(float).eps)

    return 1/m *cost_eval

def gradient_descent_log(X, y, theta_init, step_size, max_iter):
    theta_list  = [theta_init]
    m, n = X.shape
    for i in range(max_iter):
        grad = 0
        for j in range(m):
            xtmp = X[j:j +1, :]
            htheta_xtmp = sigmoid(np.squeeze(xtmp).dot(np.squeeze(theta_init)))
            grad -= (y[j] - htheta_xtmp) * xtmp.T
        
        theta_new = theta_init - step_size *grad/m
        theta_init = theta_new
        theta_list.append(theta_new)

    #Calculate error
    return np.squeeze(np.array(theta_list)), error_vals(X, y, np.squeeze(np.array(theta_list)))

def error_vals(X, y, theta_list):
    max_iter = theta_list.shape[0]
    err = []
    for i in range(max_iter):
        err_tmp = neg_log_cost(X, y, theta_list[i])
        err.append(err_tmp)
    return err

def GDA(X, y):
    m, n = X.shape

    #calculate phi
    phi = 0
    for i in range(m):
        if(y[i] == 1):
            phi += 1
    phi = phi/m

    #Calculate mu_0
    i = 0
    for i in range(m):
        if(y[i] == 0):
            mu0_num += 1*X[i]
            mu0_den += 1

    mu0 = mu0_num/mu0_den

    #Calculate mu_1
    i = 0
    for i in range(m):
        if(y[i] == 1):
            mu0_num += 1*X[i]
            mu0_den += 1    

    #Calculate capital sigma
    sigma = np.zeros(shape = (n, n))
    i = 0
    for i in range(m):
        for j in range(n):
            



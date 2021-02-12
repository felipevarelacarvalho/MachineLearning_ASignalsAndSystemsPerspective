#import libraries
import numpy as np

print("Running LinearRegression.py")

#generate data
m = 20; #number of observation
n = 10; #true data size
nois_var = 1e-8; #variance of the noise
theta_true = np.random.randn(n,1) #Made for checking answer
X = np.random.randn(m, n)
nois = (nois_var)**1/2 * np.random.randn(m, 1)
print("B", X)
y = np.dot(X, theta_true) + nois
print(theta_true)
print("A",y)

print("C", nois)
#pseudo-inverse
theta_hat_pinv = np.dot(np.linalg.pinv(X), y) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
print (np.linalg.norm(theta_hat_pinv -theta_true))

#normal equation
theta_hat_norm = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)),y)
print (np.linalg.norm(theta_hat_norm - theta_true))

#gradient descent
mu = 1e-2
max_iter = 1000
theta_hat_gd = np.zeros_like(theta_true)
for t in range(max_iter):
    theta_hat_gd = theta_hat_gd - mu *(np.dot(np.dot(np.transpose(X), X),theta_hat_gd) - np.dot(np.transpose(X),y))
    
print (np.linalg.norm(theta_hat_gd - theta_true))
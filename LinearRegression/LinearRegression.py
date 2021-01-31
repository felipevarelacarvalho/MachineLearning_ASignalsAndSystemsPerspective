#import libraries
import numpy as np

print("Hello World")

#generate data
m = 20; #number of observation
n = 10; #true data size
nois_var = 1e-8; #variance of the noise
theta_true = np.random.randn(n,1)
X = np.random.randn(m, n)
nois = (nois_var)**1/2 * np.random.randn(m, 1)

y = np.dot(X, theta_true) + nois

#pseudo-inverse
theta_hat_pinv = np.dot(np.linalg.pinv(X), y)
print (np.linalg.norm(theta_hat_pinv -theta_true))
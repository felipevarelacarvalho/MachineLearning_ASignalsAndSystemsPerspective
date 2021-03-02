#import libraries
import numpy as np
import matplotlib.pyplot as plt 
import math
from mpl_toolkits.mplot3d import Axes3D
from SyntheticDataGeneration import *
from TrainingMethods import *


#def contour_funct_lr(theta0, theta1, X, y):
#    Z = np.zeros((100, 100))
#    for i in range(100):
#        for j in range(100):
#            theta_tmp = np.array([theta0[[i], [j]], theta1[[i], [j]]])
#            Z[[i], [j]] = neg_log_cost(X, y, theta_tmp)
#    return Z

m = 20
n = 100
phi = 0.5

X, y = generate_data_classification(m, n , phi)
#print(X.shape, y.shape)

X_zero = np.array([np.array(X[i, :]) for i in range(m) if y[i] == 0])
X_one = np.array([np.array(X[i, :]) for i in range(m) if y[i] ==1])

plt.figure(figsize = (6,6))
plt.scatter(X_zero[:, 0], X_zero[:, 1], color = 'red')
plt.scatter(X_one[:, 0], X_one[:, 1], color = 'blue')
plt.title('Synthetic Generation data for classification') 
plt.show()

#--------------------------------------------------------------------------------
#xx = np.linspace(-100, 100, 100)
#yy = np.linspace(-100, 100, 100)
#theta0 , theta1 = np.meshgrid(xx, yy)
#
#Z = contour_funct_lr(theta0, theta1, X, y)
#fig = plt.figure(figsize = (6,6))
#ax = Axes3D(fig)
#ax.plot_surface(theta0, theta1, Z, cmap = 'jet')
#ax.view_init(azim = 140, elev = 10)
#plt.xlabel('theta 0')
#plt.ylabel('theta 1')
#plt.show()
#--------------------------------------------------------------------------------

max_iter = 500
step_size  = 1e-3

theta_list, cost_decay = gradient_descent_log(X, y, np.random.randn(n,1), step_size, max_iter)

plt.plot(cost_decay, 'r')
plt.yscale("log")
plt.title('iterations vs cost functions with step size 1e-3 ') 
plt.xlabel('iterations')
plt.ylabel('cost functions')
plt.show()

final_theta = theta_list[-1]
init_theta = theta_list[0]
plt.figure(figsize = (6,6))
plt.scatter(X_zero[:, 0], X_zero[:, 1], color = 'red')
plt.scatter(X_one[:, 0], X_one[:, 1], color = 'blue')
ax = plt.gca()
#ax.autoscale(False)
classifier_theta0 = np.array(ax.get_xlim())
classifier_theta1 = -(classifier_theta0 * final_theta[0])/final_theta[1]
plt.plot(classifier_theta0, classifier_theta1, '-', c="green")

classifier_init0 = np.array(ax.get_xlim())
classifier_init1 = -(classifier_init0*init_theta[0])/init_theta[1]
plt.plot(classifier_init0, classifier_init1, '-', c= "black")

plt.show()

#Estimation using real data
m_test = round(m/5)
X_test, y_test = generate_data_classification(m_test, n , phi)

h_test = np.zeros(m_test)
for i in range (m_test):
    h_test[i] = round(1/(1+ math.exp(- np.matmul(final_theta.T, X_test[i]) ) ))

print(y_test)
print(h_test)

accuracy_magnitude = 0
for ii in range(len(X_test)):
    accuracy_magnitude += np.linalg.norm(y_test - h_test[ii])
accuracy = 1/np.linalg.norm(X_test) * accuracy_magnitude
print (accuracy)

#0.07746287320564356 + 0.0866407192475633 + 0.08763492720209996 + 0.07734221937184776
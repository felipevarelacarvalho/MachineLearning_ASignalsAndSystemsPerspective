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
from MonteCarloEstimate import MonteCarloEstimate

#Parameters that we will be using for data generation
#Our data will have x ∼ N (0, I), therefore, mean zero and covariance identity
m = 80
n = 100
m_test = round(0.2*m)
m_total = m - m_test
theta = np.zeros(n)
mu = np.zeros(n)
sigma_epsilon_squared  = 0.01*( (np.linalg.norm(theta))**2 )

#θ = [100, −99, 98, −97...1]'
for i in range(n):
    if (i%2 == 0):
        theta[i] = 100 - i
    else:
        theta[i] = - 100 + i
theta = theta.transpose()

errors_m80  = MonteCarloEstimate(m_total, n, m_test, mu, theta, sigma_epsilon_squared)

m = 100
m_total = m - m_test
m_test = round(0.2*m)
errors_m100 = MonteCarloEstimate(m_total, n, m_test, mu, theta, sigma_epsilon_squared)

m = 120
m_total = m - m_test
m_test = round(0.2*m)
errors_m120 = MonteCarloEstimate(m_total, n, m_test, mu, theta, sigma_epsilon_squared)

m = 400
m_total = m - m_test
m_test = round(0.2*m)
errors_m400 = MonteCarloEstimate(m_total, n, m_test, mu, theta, sigma_epsilon_squared)


print("Monte Carlo approximation when m = 80  : ", errors_m80[0], "Estimation error when m = 80 : ", errors_m80[1])
print("Monte Carlo approximation when m = 100 : ", errors_m100[0], "Estimation error when m = 100 : ", errors_m100[1])
print("Monte Carlo approximation when m = 120 : ", errors_m120[0], "Estimation error when m = 120 : ", errors_m120[1])
print("Monte Carlo approximation when m = 400 : ", errors_m400[0], "Estimation error when m = 400 : ", errors_m400[1])
plot = plt.gca();
plot.scatter(80, errors_m80[0], label = "part_a m = 80, Monte Carlo");
plot.annotate(errors_m80[0], (80, errors_m80[0]))
plot.scatter(100, errors_m100[0], label = "part_a m = 100, Monte Carlo");
plot.annotate(errors_m100[0], (100, errors_m100[0]))
plot.scatter(120, errors_m120[0], label = "part_a m = 120, Monte Carlo");
plot.annotate(errors_m120[0], (120, errors_m120[0]))
plot.scatter(400, errors_m400[0], label = "part_a m = 400, Monte Carlo");
plot.annotate(errors_m400[0], (400, errors_m400[0]))
# naming the x axis 
plt.xlabel('m values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Monte Carlo error, sigma squared = 0.01 norm squared of theta') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 

plot = plt.gca();
plot.scatter(80, errors_m80[1], label = "part_a m = 80, estimation error theta");
plot.annotate(errors_m80[1], (80, errors_m80[1]))
plot.scatter(100, errors_m100[1], label = "part_a m = 100, estimation error theta");
plot.annotate(errors_m100[1], (100, errors_m100[1]))
plot.scatter(120, errors_m120[1], label = "part_a m = 120, estimation error theta");
plot.annotate(errors_m120[1], (120, errors_m120[1]))
plot.scatter(400, errors_m400[1], label = "part_a m = 400, estimation error theta");
plot.annotate(errors_m400[1], (400, errors_m400[1]))
# naming the x axis 
plt.xlabel('m values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Estimation error, sigma squared = 0.01 norm squared of theta') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 

print("\n")
#Repeat process with new sigma
#-------------------------------------------------------------------------------------------
m = 80
n = 100
m_test = round(0.2*m)
m_total = m - m_test
theta = np.zeros(n)
mu = np.zeros(n)
sigma_epsilon_squared  = 0.1*( (np.linalg.norm(theta))**2 )

#θ = [100, −99, 98, −97...1]'
for i in range(n):
    if (i%2 == 0):
        theta[i] = 100 - i
    else:
        theta[i] = - 100 + i
theta = theta.transpose()

errors_m80  = MonteCarloEstimate(m, n, m_test, mu, theta, sigma_epsilon_squared)

m = 100
m_total = m - m_test
m_test = round(0.2*m)
errors_m100 = MonteCarloEstimate(m_total, n, m_test, mu, theta, sigma_epsilon_squared)

m = 120
m_total = m - m_test
m_test = round(0.2*m)
errors_m120 = MonteCarloEstimate(m_total, n, m_test, mu, theta, sigma_epsilon_squared)

m = 400
m_total = m - m_test
m_test = round(0.2*m)
errors_m400 = MonteCarloEstimate(m_total, n, m_test, mu, theta, sigma_epsilon_squared)


print("Monte Carlo approximation when m = 80  : ", errors_m80[0], "Estimation error when m = 80 : ", errors_m80[1])
print("Monte Carlo approximation when m = 100 : ", errors_m100[0], "Estimation error when m = 100 : ", errors_m100[1])
print("Monte Carlo approximation when m = 120 : ", errors_m120[0], "Estimation error when m = 120 : ", errors_m120[1])
print("Monte Carlo approximation when m = 400 : ", errors_m400[0], "Estimation error when m = 400 : ", errors_m400[1])
plot = plt.gca();
plot.scatter(80, errors_m80[0], label = "part_a m = 80, Monte Carlo");
plot.annotate(errors_m80[0], (80, errors_m80[0]))
plot.scatter(100, errors_m100[0], label = "part_a m = 100, Monte Carlo");
plot.annotate(errors_m100[0], (100, errors_m100[0]))
plot.scatter(120, errors_m120[0], label = "part_a m = 120, Monte Carlo");
plot.annotate(errors_m120[0], (120, errors_m120[0]))
plot.scatter(400, errors_m400[0], label = "part_a m = 400, Monte Carlo");
plot.annotate(errors_m400[0], (400, errors_m400[0]))
# naming the x axis 
plt.xlabel('m values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Monte Carlo error, sigma squared = 0.1 norm squared of theta') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 

plot = plt.gca();
plot.scatter(80, errors_m80[1], label = "part_a m = 80, estimation error theta");
plot.annotate(errors_m80[1], (80, errors_m80[1]))
plot.scatter(100, errors_m100[1], label = "part_a m = 100, estimation error theta");
plot.annotate(errors_m100[1], (100, errors_m100[1]))
plot.scatter(120, errors_m120[1], label = "part_a m = 120, estimation error theta");
plot.annotate(errors_m120[1], (120, errors_m120[1]))
plot.scatter(400, errors_m400[1], label = "part_a m = 400, estimation error theta");
plot.annotate(errors_m400[1], (400, errors_m400[1]))
# naming the x axis 
plt.xlabel('m values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Estimation error, sigma squared = 0.1 norm squared of theta') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 


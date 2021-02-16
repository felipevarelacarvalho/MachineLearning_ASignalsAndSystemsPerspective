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

#generate test data
#SimulateSyntheticData(m, n, mu, covariance, theta)
#m, n and covariance : int
#mu : array
#theta : array
#returns array[x_array, error_array, y_array, x_array_true, error_array_true, y_array_true]

start = timeit.default_timer() #Starts  timer  
part_a_1 = SimulateSyntheticData(30, 5, [0], 0,  np.array([[1], [4], [2], [10], [23]]) )
stop = timeit.default_timer() #Stops timer
print('Time part a) in seconds  m= 30 : ', stop - start) #prints time

start = timeit.default_timer() #Starts  timer  
part_a_2 = SimulateSyntheticData(100, 5, [0], 0,  np.array([[1], [4], [2], [10], [23]]) )
stop = timeit.default_timer() #Stops timer
print('Time part a) in seconds m= 100 : ', stop - start) #prints time

start = timeit.default_timer() #Starts  timer  
part_a_3 = SimulateSyntheticData(1000, 5, [0], 0,  np.array([[1], [4], [2], [10], [23]]) )
stop = timeit.default_timer() #Stops timer
print('Time part a) in seconds m= 1000 : ', stop - start) #prints time

start = timeit.default_timer() #Starts  timer  
part_b_1 = SimulateSyntheticData(30, 5, [0], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )
stop = timeit.default_timer() #Stops timer
print('Time part b) in seconds m= 30 : ', stop - start) #prints time

start = timeit.default_timer() #Starts  timer  
part_b_2 = SimulateSyntheticData(100, 5, [0], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )
stop = timeit.default_timer() #Stops timer
print('Time part b) in seconds m= 100 : ', stop - start) #prints time

start = timeit.default_timer() #Starts  timer  
part_b_3 = SimulateSyntheticData(1000, 5, [0], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )
stop = timeit.default_timer() #Stops timer
print('Time part b) in seconds m= 1000 : ', stop - start) #prints time

start = timeit.default_timer() #Starts  timer  
part_c_1 = SimulateSyntheticData(30, 5, [0], 10**(-4),  np.array([[1], [4], [2], [10], [23]]) )
stop = timeit.default_timer() #Stops timer
print('Time part c) in seconds m= 30 : ', stop - start) #prints time

start = timeit.default_timer() #Starts  timer  
part_c_2 = SimulateSyntheticData(100, 5, [0], 10**(-4),  np.array([[1], [4], [2], [10], [23]]) )
stop = timeit.default_timer() #Stops timer
print('Time part c) in seconds m= 100 : ', stop - start) #prints time

start = timeit.default_timer() #Starts  timer  
part_c_3 = SimulateSyntheticData(1000, 5, [0], 10**(-4),  np.array([[1], [4], [2], [10], [23]]) )
stop = timeit.default_timer() #Stops timer
print('Time part c) in seconds m= 1000 : ', stop - start) #prints time
print("\n")

#Learning the value of theta
theta_array = np.array([[1], [4], [2], [10], [23]])
#pseudo-inverse
#For a)
pseudo_inverse = PseudoInverse(part_a_1[0], part_a_1[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_a1 = pseudo_inverse[1]
print ("Pseudo Inverse normalization error for a) m = 30 : ", pseudo_inverse_error_a1 )

pseudo_inverse = PseudoInverse(part_a_2[0], part_a_2[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_a2 = pseudo_inverse[1]
print ("Pseudo Inverse normalization error for a) m = 100 : ", pseudo_inverse_error_a2 )

pseudo_inverse = PseudoInverse(part_a_3[0], part_a_3[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_a3 = pseudo_inverse[1]
print ("Pseudo Inverse normalization error for a) m = 1000 : ",pseudo_inverse_error_a3 )
#For b)
pseudo_inverse = PseudoInverse(part_b_1[0], part_b_1[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_b1 = pseudo_inverse[1]
print ("Pseudo Inverse normalization error for b) m = 30 : ",pseudo_inverse_error_b1 )

pseudo_inverse = PseudoInverse(part_b_2[0], part_b_2[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_b2 = pseudo_inverse[1]
print ("Pseudo Inverse normalization error for b) m = 100 : ", pseudo_inverse_error_b2)

pseudo_inverse = PseudoInverse(part_b_3[0], part_b_3[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_b3 = pseudo_inverse[1]
print ("Pseudo Inverse normalization error for b) m = 1000 : ", pseudo_inverse_error_b3)
print("\n")

#normal equation
#for a)
normal_Equation = NormalEquation(part_a_1[0], part_a_1[2], theta_array )
normal_error_a1 = normal_Equation[1]
print("Normal Equation normalized error for a) m = 30 : ", normal_error_a1)

normal_Equation = NormalEquation(part_a_2[0], part_a_2[2], theta_array )
normal_error_a2 = normal_Equation[1]
print("Normal Equation normalized error for a) m = 100 : ", normal_error_a2)

normal_Equation = NormalEquation(part_a_3[0], part_a_3[2], theta_array )
normal_error_a3 = normal_Equation[1]
print("Normal Equation normalized error for a) m = 1000 : ", normal_error_a3)
#for b)
normal_Equation = NormalEquation(part_b_1[0], part_b_1[2], theta_array )
normal_error_b1 = normal_Equation[1]
print("Normal Equation normalized error for b) m = 30 : ", normal_error_b1)

normal_Equation = NormalEquation(part_b_2[0], part_b_2[2], theta_array )
normal_error_b2 = normal_Equation[1]
print("Normal Equation normalized error for b) m = 100 : ", normal_error_b2)

normal_Equation = NormalEquation(part_b_3[0], part_b_3[2], theta_array )
normal_error_b3 = normal_Equation[1]
print("Normal Equation normalized error for b) m = 1000 : ", normal_error_b3)
print("\n")

#Gradient Descent
#for a)
gradient_descent = GradientDescent(part_a_1[0], part_a_1[2], theta_array, 0.01, 1000 )
gradient_error_a1 = gradient_descent[1]
print("Gradient Descent normalized error for a) m = 30 : ", gradient_error_a1)

gradient_descent = GradientDescent(part_a_2[0], part_a_2[2], theta_array, 0.001, 1000 )
gradient_error_a2 = gradient_descent[1]
print("Gradient Descent normalized error for a) m = 100 : ", gradient_error_a2)

gradient_descent = GradientDescent(part_a_3[0], part_a_3[2], theta_array, 0.001, 1000 )
gradient_error_a3 = gradient_descent[1]
print("Gradient Descent normalized error for a) m = 1000 : ", gradient_error_a3)

#for b)
gradient_descent = GradientDescent(part_b_1[0], part_b_1[2], theta_array, 0.01, 1000 )
gradient_error_b1 = gradient_descent[1]
print("Gradient Descent normalized error for b) m = 30 : ", gradient_error_b1)

gradient_descent = GradientDescent(part_b_2[0], part_b_2[2], theta_array, 0.001, 1000 )
gradient_error_b2 = gradient_descent[1]
print("Gradient Descent normalized error for b) m = 100 : ", gradient_error_b2)

gradient_descent = GradientDescent(part_b_3[0], part_b_3[2], theta_array, 0.001, 1000 )
gradient_error_b3 = gradient_descent[1]
print("Gradient Descent normalized error for b) m = 1000 : ", gradient_error_b3)

#Plot
plot = plt.gca();
plot.scatter(30, pseudo_inverse_error_a1, label = "part_a m = 30, Pseudo-Inverse");
plot.scatter(30, normal_error_a1, label = "part_a m = 30, Normal-Equation");
plot.scatter(30, gradient_error_a1, label = "part_a m = 30, Grandient-Descent");
plot.scatter(100, pseudo_inverse_error_a2, label = "part_a m = 100, Pseudo-Inverse");
plot.scatter(100, normal_error_a2, label = "part_a m = 100, Normal-Equation");
plot.scatter(100, gradient_error_a2, label = "part_a m = 100, Grandient-Descent");
plot.scatter(1000, pseudo_inverse_error_a3, label = "part_a m = 1000, Pseudo-Inverse");
plot.scatter(1000, normal_error_a3, label = "part_a m = 1000, Normal-Equation");
plot.scatter(1000, gradient_error_a3, label = "part_a m = 1000, Grandient-Descent");
# naming the x axis 
plt.xlabel('m values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Part 1 results for a') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 

plot = plt.gca();
plot.scatter(30, pseudo_inverse_error_b1, label = "part_b m =30, Pseudo-Inverse");
plot.scatter(30, normal_error_b1, label = "part_b m =30, Normal-Equation");
plot.scatter(30, gradient_error_b1, label = "part_a m = 30, Grandient-Descent");
plot.scatter(100, pseudo_inverse_error_b2, label = "part_b m =100, Pseudo-Inverse");
plot.scatter(100, normal_error_b2, label = "part_b m =100, Normal-Equation");
plot.scatter(100, gradient_error_b2, label = "part_a m = 100, Grandient-Descent");
plot.scatter(1000, pseudo_inverse_error_b3, label = "part_b m =1000, Pseudo-Inverse");
plot.scatter(1000, normal_error_b3, label = "part_b m =1000, Normal-Equation");
plot.scatter(1000, gradient_error_b3, label = "part_a m = 1000, Grandient-Descent");
# naming the x axis 
plt.xlabel('m values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Part 1 results for b') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 

#----------------------
#Fixing the learning on Pseudo-Inverse
pseudo_inverse = PseudoInverse(part_a_1[0], part_a_1[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_a1 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_a_2[0], part_a_2[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_a2 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_a_3[0], part_a_3[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_a3 = pseudo_inverse[1]

pseudo_inverse = PseudoInverse(part_b_1[0], part_b_1[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_b1 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_b_2[0], part_b_2[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_b2 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_b_3[0], part_b_3[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_b3 = pseudo_inverse[1]

pseudo_inverse = PseudoInverse(part_c_1[0], part_c_1[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_c1 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_c_2[0], part_c_2[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_c2 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_c_3[0], part_c_3[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_c3 = pseudo_inverse[1]

#Plot
plot = plt.gca();
plot.scatter(30, pseudo_inverse_error_a1, label = "part_a m = 30");
#plot.annotate('part_a m = 30', (30, pseudo_inverse_error_a1))
plot.scatter(30, pseudo_inverse_error_b1, label = "part_b m = 30");
#plot.annotate('part_b m = 30', (30, pseudo_inverse_error_b1))
plot.scatter(30, pseudo_inverse_error_c1, label = "part_c m = 30");
#plot.annotate('part_c m = 30', (30, pseudo_inverse_error_c1))
plot.scatter(100, pseudo_inverse_error_a2, label = "part_a m = 100");
#plot.annotate('part_a m = 100', (100, pseudo_inverse_error_a2))
plot.scatter(100, pseudo_inverse_error_b2, label = "part_b m = 100");
#plot.annotate('part_b m = 100', (100, pseudo_inverse_error_b2))
plot.scatter(100, pseudo_inverse_error_c2, label = "part_c m = 100");
#plot.annotate('part_c m = 100', (100, pseudo_inverse_error_c2))
plot.scatter(1000, pseudo_inverse_error_a3, label = "part_a m = 1000");
#plot.annotate('part_a m = 1000', (1000, pseudo_inverse_error_a3))
plot.scatter(1000, pseudo_inverse_error_b3, label = "part_b m = 1000");
#plot.annotate('part_b m = 1000', (1000, pseudo_inverse_error_b3))
plot.scatter(1000, pseudo_inverse_error_c3, label = "part_c m = 1000");
#plot.annotate('part_c m = 1000', (1000, pseudo_inverse_error_c3))
# naming the x axis 
plt.xlabel('m values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Part 2 Pseudo Inverse results a) b) and c)') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 

#-----------------------------
print("-----------------------------------\n")
print("Training a data set witha non zero mean:\n")
part_d_1 = SimulateSytheticNonZeroMean(30, 5, [1,1,1,1,1], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )
part_d_2 = SimulateSytheticNonZeroMean(100, 5, [1,1,1,1,1], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )
part_d_3 = SimulateSytheticNonZeroMean(1000, 5, [1,1,1,1,1], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )

print("Training Using Pseudo Inverse with mean parameter theta_not:")
pseudo_inverse = PseudoInverse(part_d_1[0], part_d_1[2], part_d_1[3] ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_d1 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_d_2[0], part_d_2[2], part_d_2[3] ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_d2 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_d_3[0], part_d_3[2], part_d_3[3] ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_d3 = pseudo_inverse[1]

print ("Pseudo Inverse normalization error for d) using mean param. m = 30 : ", pseudo_inverse_error_d1 )
print ("Pseudo Inverse normalization error for d) using mean param. m = 100 : ", pseudo_inverse_error_d2 )
print ("Pseudo Inverse normalization error for d) using mean param. m = 1000 : ", pseudo_inverse_error_d3 )

print("\nTraining Using Pseudo Inverse without mean parameter theta_not:")
part_d_1 = SimulateSyntheticData(30, 5, [1,1,1,1,1], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )
part_d_2 = SimulateSyntheticData(100, 5, [1,1,1,1,1], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )
part_d_3 = SimulateSyntheticData(1000, 5, [1,1,1,1,1], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )

pseudo_inverse = PseudoInverse(part_d_1[0], part_d_1[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_d1 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_d_2[0], part_d_2[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_d2 = pseudo_inverse[1]
pseudo_inverse = PseudoInverse(part_d_3[0], part_d_3[2], theta_array ) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
pseudo_inverse_error_d3 = pseudo_inverse[1]

print ("Pseudo Inverse normalization error for d) using mean param. m = 30 : ", pseudo_inverse_error_d1 )
print ("Pseudo Inverse normalization error for d) using mean param. m = 100 : ", pseudo_inverse_error_d2 )
print ("Pseudo Inverse normalization error for d) using mean param. m = 1000 : ", pseudo_inverse_error_d3 )

part_e_1 = SimulateSytheticDiagonalCovariance(30, 5, [0,0], [[3, 0], [0, 30]], 10**(-6), np.array([[1, 4, 2, 10, 23], [1, 4, 2, 10, 23]]) )
part_e_2 = SimulateSytheticDiagonalCovariance(30, 5, [0,0], [[3, 0], [0, 30]], 10**(-6), np.array([[1, 4, 2, 10, 23], [1, 4, 2, 10, 23]]) )
part_e_3 = SimulateSytheticDiagonalCovariance(30, 5, [0,0], [[3, 0], [0, 30]], 10**(-6), np.array([[1, 4, 2, 10, 23], [1, 4, 2, 10, 23]]) )

print("\nSimulated data from e) y=") 
print(part_e_1[0])
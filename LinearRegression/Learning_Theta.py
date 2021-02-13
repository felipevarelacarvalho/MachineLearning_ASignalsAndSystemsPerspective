#import libraries
import numpy as np
import matplotlib.pyplot as plt 
from SimulatedSyntheticData import SimulateSyntheticData
from TrainingMethods import PseudoInverse
from TrainingMethods import NormalEquation
from TrainingMethods import GradientDescent

#generate test data
#SimulateSyntheticData(m, n, mu, covariance, theta)
#m, n and covariance : int
#mu : array
#theta : array
#returns array[x_array, error_array, y_array, x_array_true, error_array_true, y_array_true]
part_a_1 = SimulateSyntheticData(30, 5, [0], 0,  np.array([[1], [4], [2], [10], [23]]) )
part_a_2 = SimulateSyntheticData(100, 5, [0], 0,  np.array([[1], [4], [2], [10], [23]]) )
part_a_3 = SimulateSyntheticData(1000, 5, [0], 0,  np.array([[1], [4], [2], [10], [23]]) )

part_b_1 = SimulateSyntheticData(30, 5, [0], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )
part_b_2 = SimulateSyntheticData(100, 5, [0], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )
part_b_3 = SimulateSyntheticData(1000, 5, [0], 10**(-6),  np.array([[1], [4], [2], [10], [23]]) )

part_c_1 = SimulateSyntheticData(30, 5, [0], 10**(-4),  np.array([[1], [4], [2], [10], [23]]) )
part_c_2 = SimulateSyntheticData(100, 5, [0], 10**(-4),  np.array([[1], [4], [2], [10], [23]]) )
part_c_3 = SimulateSyntheticData(1000, 5, [0], 10**(-4),  np.array([[1], [4], [2], [10], [23]]) )

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

normal_Equation = NormalEquation(part_b_1[0], part_b_1[2], theta_array )
normal_error_b1 = normal_Equation[1]
print("Normal Equation normalized error for b) m = 30 : ", normal_error_b1)

normal_Equation = NormalEquation(part_b_2[0], part_b_2[2], theta_array )
normal_error_b2 = normal_Equation[1]
print("Normal Equation normalized error for b) m = 100 : ", normal_error_b2)

normal_Equation = NormalEquation(part_b_3[0], part_b_3[2], theta_array )
normal_error_b3 = normal_Equation[1]
print("Normal Equation normalized error for b) m = 1000 : ", normal_error_b3)

#Plot
plot = plt.gca();
plot.scatter(30, pseudo_inverse_error_a1, label = "part_a_1");
plot.scatter(100, pseudo_inverse_error_a2, label = "part_a_2");
plot.scatter(1000, pseudo_inverse_error_a3, label = "part_a_3");
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
plot.scatter(30, pseudo_inverse_error_b1, label = "part_b_1");
plot.scatter(100, pseudo_inverse_error_b2, label = "part_b_2");
plot.scatter(1000, pseudo_inverse_error_b3, label = "part_b_3");
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

#TODO 
#Make plor for Normal Equation
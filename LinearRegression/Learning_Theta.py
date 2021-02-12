#import libraries
import numpy as np
import matplotlib.pyplot as plt 
from SimulatedSyntheticData import SimulateSyntheticData

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

#pseudo-inverse
#For a)
theta_hat_pinv = np.dot(np.linalg.pinv(part_a_1[0]), part_a_1[2]) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
nomalized_error_a1 = np.linalg.norm(theta_hat_pinv - np.array([[1], [4], [2], [10], [23]]))
print ("Pseudo Inverse normalization error for a) m = 30 : ", nomalized_error_a1 )

theta_hat_pinv = np.dot(np.linalg.pinv(part_a_2[0]), part_a_2[2]) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
nomalized_error_a2 = np.linalg.norm(theta_hat_pinv - np.array([[1], [4], [2], [10], [23]]))
print ("Pseudo Inverse normalization error for a) m = 100 : ", nomalized_error_a2 )

theta_hat_pinv = np.dot(np.linalg.pinv(part_a_3[0]), part_a_3[2]) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
nomalized_error_a3 = np.linalg.norm(theta_hat_pinv - np.array([[1], [4], [2], [10], [23]]))
print ("Pseudo Inverse normalization error for a) m = 1000 : ",nomalized_error_a3 )
print("\n")
#For b)
theta_hat_pinv = np.dot(np.linalg.pinv(part_b_1[0]), part_b_1[2]) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
nomalized_error_b1 = np.linalg.norm(theta_hat_pinv - np.array([[1], [4], [2], [10], [23]]))
print ("Pseudo Inverse normalization error for b) m = 30 : ",nomalized_error_b1 )

theta_hat_pinv = np.dot(np.linalg.pinv(part_b_2[0]), part_b_2[2]) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
nomalized_error_b2 = np.linalg.norm(theta_hat_pinv - np.array([[1], [4], [2], [10], [23]]))
print ("Pseudo Inverse normalization error for b) m = 100 : ", nomalized_error_b2)

theta_hat_pinv = np.dot(np.linalg.pinv(part_b_3[0]), part_b_3[2]) #Pseudo Inverse is = (A_transpose A)_Inverse)A_transpose
nomalized_error_b3 = np.linalg.norm(theta_hat_pinv - np.array([[1], [4], [2], [10], [23]]))
print ("Pseudo Inverse normalization error for b) m = 1000 : ", nomalized_error_b3)

#Plot
plot = plt.gca();
plot.scatter(30, part_a_1[1]);
# naming the x axis 
plt.xlabel('m values') 
# naming the y axis 
plt.ylabel('error') 
# giving a title to my graph 
plt.title('Part 1 results') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show() 
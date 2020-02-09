# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:47:49 2020

@author: Praveen
"""

import numpy as np
import matplotlib.pyplot as plt

'''
def leastSquares(x, y):
    x_squared = np.power(x,2)
    x_matrix = np.transpose([x_squared, x, np.ones(np.shape(x))])
    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_matrix), x_matrix)),
                  np.matmul(np.transpose(x_matrix), y))
    # print(B)
    y_estimate = np.matmul(x_matrix, np.transpose(B))
    # plt.scatter(x_matrix[:,1], y, label = 'Scattered data')
    # plt.plot(x_matrix[:,1], y_estimate, 'r', label = 'Curve Fitting')
    return B
'''
def leastSquares(x, y):
     x_squared = np.power(x,2)
     x_matrix = np.transpose([x_squared, x, np.ones(np.shape(x))])
     B = np.dot(np.linalg.inv(x_matrix), np.transpose(y))
     return B
 
def hypothesisEvaluate(x_matrix, y, hypothesis_parameters, distance_threshold):
    a = 4*hypothesis_parameters[0]
    b = 6*hypothesis_parameters[0]*hypothesis_parameters[1]
    x0 = np.reshape(x_matrix[:,1],(np.shape(y)[0],1))
    count = 0
    distance = []
    for i in range(np.shape(y)[0]):
        c = 2*(1 + 2*hypothesis_parameters[0]*(hypothesis_parameters[2] - y[i]) + hypothesis_parameters[1])
        d = 2*(hypothesis_parameters[1]*(-y[i] + hypothesis_parameters[2]) - x0[i])
        soln = np.real(np.roots([a, b, c, d]))
        soln_squared = np.power(soln,2)
        soln_matrix = np.transpose([soln_squared, soln, np.ones(np.shape(soln))])
        ymin = np.matmul(soln_matrix, np.transpose(hypothesis_parameters))
        distance_soln = np.sqrt(np.power((soln-x0[i]),2) + np.power((ymin-y[i]),2))
        distance.append(np.amin(distance_soln))
        
    print(distance)
    inliers = np.array(distance) <= distance_threshold
    count = np.count_nonzero(inliers)
    return count

'''
def hypothesisEvaluate(x_matrix, y, hypothesis_parameters, distance_threshold):
    # y_estimate = np.matmul(x_matrix, np.transpose(hypothesis_parameters))
    constant = (8*hypothesis_parameters[0]*hypothesis_parameters[0] + 2)
    K = 2/constant
    L = 4*hypothesis_parameters[0]/constant
    C = 4*hypothesis_parameters[0]*hypothesis_parameters[1]/constant
    x0 = np.reshape(x_matrix[:,1],(np.shape(y)[0],1))
    y0 = np.reshape(y,(np.shape(y)[0],1))
    X = np.dot(x0, K) + np.dot(y0, L) + np.dot(np.ones((np.shape(x_matrix)[0],1)),C)
    distance = np.sqrt(np.power((X-x0),2) + np.power((np.dot(X,2*hypothesis_parameters[0]) + np.dot(np.ones((np.shape(x_matrix)[0],1)),hypothesis_parameters[1]) - y0),2))
    count = 0
    inliers = distance <= distance_threshold
    count = np.count_nonzero(inliers)
    return count
'''    

def RANSAC(x_matrix, y, minimum_no_of_inliers, distance_threshold, iterations, init_points):
    total_data_points = np.shape(x_matrix)[0]
    x = x_matrix[:,1]
    best_parameters = []
    best_inliers = 0
    i = 0
    while i <= iterations:
        min_points_indices = np.random.choice(total_data_points, init_points, replace = False)
        hypothesis_parameters = leastSquares(x[min_points_indices], y[min_points_indices])
        number_of_inliers = hypothesisEvaluate(x_matrix, y, hypothesis_parameters, distance_threshold)
        if number_of_inliers >= minimum_no_of_inliers:
            best_parameters = hypothesis_parameters
            best_inliers = number_of_inliers
        i = i + 1
    
    return [best_parameters, best_inliers]

def main():
    curve_data = np.genfromtxt('data_2.csv', delimiter = ',')
    x = curve_data[1:,0]
    y = curve_data[1:,1]
    x_squared = np.power(x,2)
    x_matrix = np.transpose([x_squared, x, np.ones(np.shape(x))])
    parameters = RANSAC(x_matrix, y, 50, 5, 10, 3)
    y_estimate = np.matmul(x_matrix, np.transpose(parameters[0]))
    plt.scatter(x, y, label = 'Scattered data')
    plt.plot(x, y_estimate, 'r', label = 'Curve Fitting')
    
if __name__ == '__main__':
    main()
        
        
        
        
        
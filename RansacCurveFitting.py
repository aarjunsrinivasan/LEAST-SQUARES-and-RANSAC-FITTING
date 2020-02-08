# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:47:49 2020

@author: Praveen
"""

import numpy as np

def leastSquares(x, y):
    '''
    Evaluate model for the 3 points chosen and return model parameters
    '''
    x_squared = np.power(x,2)
    x_matrix = np.transpose([x_squared, x, np.ones(np.shape(x))])
    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_matrix), x_matrix)),
                  np.matmul(np.transpose(x_matrix), y))
    return B

def hypothesisEvaluate(x, y, distance_threshold, hypothesis_parameters):
'''    
    Find the number of inliers for the current model and return it
'''

def RANSAC(x, y, minimum_no_of_inliers, distance_threshold, iterations, init_points):
    total_data_points = np.shape(x)[0]
    i = 0
    while i <= iterations:
        min_points_indices = np.random.choice(total_data_points, init_points, replace = False)
        hypothesis_parameters = leastSquares(x[min_points_indices], y[min_points_indices])
        number_of_inliers = hypothesisEvaluate(x, y, distance_threshold, hypothesis_parameters)
        i = i + 1
    
    return hypothesis_parameters

def main():
    curve_data = np.genfromtxt('data_2.csv', delimiter = ',')
    x = curve_data[1:,0]
    y = curve_data[1:,1]
    parameters = RANSAC(x, y, 0, 0, 1, 3)
    print(parameters)
    
if __name__ == '__main__':
    main()
        
        
        
        
        
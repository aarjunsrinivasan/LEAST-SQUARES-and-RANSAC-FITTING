# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:01:05 2020

@author: Arun
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:32:15 2020
@author: Praveen
"""

import numpy as np
import matplotlib.pyplot as plt

def leastSquaresCurveFit(min_points):
    '''
    Function to fit a curve using Least squares given minimum number of points (x,y) from the data.
    Y = XB
    E = (Y - XB)T(Y - XB)
    dE/dB = 0
    B = inv(XTX)*XT*Y 
    Input ---> min_points: minimum number of points (x,y) to fit a given curve
    Output ---> B: return model parametersnfor a given curve
    '''
    # x coordinates of the min number of points required to model the given curve
    min_points_x = np.reshape(min_points[:,0], (min_points.shape[0], 1))
    # y coordinates of the min number of points required to model the given curve
    min_points_y = np.reshape(min_points[:,1], (min_points.shape[0], 1))
    min_points_x_squared = np.power(min_points_x,2)
    # Form the X matrix; for a quadratic curve it is [x^2, x, 1]
    min_points_x_matrix = np.hstack(( min_points_x_squared, min_points_x, np.ones((min_points_x.shape[0],1))))
    # Evaluate the model parameters
    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(min_points_x_matrix), min_points_x_matrix)), 
                  np.matmul(np.transpose(min_points_x_matrix), min_points_y))
    
    return B

def hypothesisEvaluate(remaining_points, hypothesis_parameters, threshold, remaining_points_indices, flag = 0):
    '''
    Funtion to evaluate/test the hypothesis or model that was used to fit the curve from minimum number of points (x,y).
    Input ---> remaining_points: points other than the ones used to find initial model parameters
               hypothesis_parameters: model parameters
               threshold: accepted error to consider a point as inlier or outlier
               remaining_points_indices: index corresponding to remaining_points
               flag: value that chooses whether to use remaining_points_indices or not
    Output ---> error: difference between estimated y and ground truth
                inliers_indices: index corresponding to inliers
    '''
    # x coordinates of the remaining points of the data required to model the given curve
    remaining_points_x = np.reshape(remaining_points[:,0], (remaining_points.shape[0], 1))
    # y coordinates of the remaining points of the data required to model the given curve
    remaining_points_y = np.reshape(remaining_points[:,1], (remaining_points.shape[0], 1))
    remaining_points_x_squared = np.power(remaining_points_x,2)
    # Form the X matrix; for a quadratic curve it is [x^2, x, 1]
    remaining_points_x_matrix = np.hstack((remaining_points_x_squared, remaining_points_x, np.ones((remaining_points.shape[0],1))))
    # Estimate y coordinate of remaining points given x coordinates and the model parameters
    y_estimate = np.dot(remaining_points_x_matrix, hypothesis_parameters)
    # Calculate the error between the estimated y and ground truth y
    error = abs(y_estimate - remaining_points_y)
    # Find the inliers; points for which error is less than or equal to acceptable threshold
    inliers = error <= threshold
    if flag == 0:
        remaining_points_indices = np.reshape(remaining_points_indices, (remaining_points_indices.shape[0],1))
        # Obtain the indices of the inlier points
        inliers_indices = remaining_points_indices[inliers]
    else:
       inliers_indices = []
       
    return error, inliers_indices
    

def RANSAC(curve, minimum_no_of_inliers, threshold, iterations, init_points):
    '''
    Function to implement the RANSAC algorithm.
    
    For i <= iterations:
        1. randomly choose n minimum points to fit the curve. For eg., to fit a line we need 2 minimum points,
           whereas to fit a quadratic curve we need 3 minimum points.
        2. fit a curve using n minimum points and obtain model parameters.
        3. evaluate the model parameters, calculate the error and inliers.
        4. if inliers >= minimum number of inliers the model should have
            1. find a model that fits the curve for entire data
            2. evaluate the model once again to find the best model
            
    Input ---> curve: (x,y) points of the curve data
               minimum_no_of_inliers: minimum number of inliers that a model should have
               threshold: acceptable error for a point to be considered as inlier
               iterations: Number of iterations the algorithm has to run
               init_points: minimum number of points (x,y) from the data required to fit a curve
    '''
    # Initialize best model parameters
    best_parameters = None
    # Initialize best inliers 
    best_inliers = 0
    # Initialize model error
    init_error = np.inf
    i = 0
    while i <= iterations:
        all_point_indices = np.arange(curve.shape[0])
        np.random.shuffle(all_point_indices)
        # Obtain indices of minimum number of points required to model a given curve
        min_points_indices = all_point_indices[:init_points]
        # Obtain indices of remaining points in the data
        remaining_points_indices = all_point_indices[init_points:]
        min_points = curve[min_points_indices,:]
        remaining_points = curve[remaining_points_indices]
        # Obtain the hypothesis/model parameters given the minimum number of points from the data
        hypothesis_parameters = leastSquaresCurveFit(min_points)
        # Evaluate the obtained hypothesis/model parameters and calculate the error, inliers
        error, inliers_indices = hypothesisEvaluate(remaining_points, hypothesis_parameters, threshold, remaining_points_indices)
        inliers = curve[inliers_indices,:]
        if len(inliers) >= minimum_no_of_inliers:
            total_inliers = np.concatenate((min_points, inliers))
            current_hypothesis_parameters = leastSquaresCurveFit(total_inliers)
            current_error, current_inliers_indices = hypothesisEvaluate(total_inliers, current_hypothesis_parameters, threshold, remaining_points_indices, 1)
            current_error = np.mean(current_error)
            if current_error <= init_error:
                best_parameters = current_hypothesis_parameters
                best_inliers = np.concatenate((min_points_indices, inliers_indices))
                init_error = current_error
        i = i + 1
    
    if np.any(best_parameters == None):
        raise ValueError("No solution")
    else:
        return best_parameters, best_inliers

def main():
    # Load curve data from csv files
    curve_data = np.genfromtxt('data_2.csv', delimiter = ',')
    x = curve_data[1:,0]
    y = curve_data[1:,1]
    x = np.reshape(x, (np.shape(x)[0], 1))
    y = np.reshape(y, (np.shape(y)[0], 1))
    curve = np.hstack((x,y))
    parameters = RANSAC(curve, 125, 20, 100, 3) 
    x_squared = np.power(x,2)
    x_matrix = np.hstack(( x_squared, x, np.ones((x.shape[0],1))))
    y_estimate = np.matmul(x_matrix, parameters[0])
    # Plot curve data, inlier data, curve estimate
    plt.figure()
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Data_2.Paramters:Probability of inliners=125,Error Threshold=20,Iterations=100')
    plt.scatter(x, y, label = 'Scattered data')
    plt.plot(x[parameters[1]], y[parameters[1]], 'bx',color='yellow', label = 'Inlier Data')
    plt.plot(x, y_estimate, 'r', label = 'Curve Fitting')
    plt.legend()
    
    
if __name__ == '__main__':
    main()
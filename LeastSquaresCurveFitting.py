# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:04:08 2020

@author: Praveen
"""

import numpy as np
import matplotlib.pyplot as plt

def leastSquares_curveFitting():
    '''
    Function to fit a curve using Least squares given minimum number of points (x,y) from the data.
    Y = XB
    E = (Y - XB)T(Y - XB)
    dE/dB = 0
    B = inv(XTX)*XT*Y 
    '''
    # Load curve data from csv files
    curve_data = np.genfromtxt('data_2.csv', delimiter = ',')
     # Extract x and y coordinates from the curve data
    x = curve_data[1:,0]
    y = curve_data[1:,1]
    x_squared = np.power(x,2)
    x_matrix = np.transpose([x_squared, x, np.ones(np.shape(x))])
    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_matrix), x_matrix)),
                  np.matmul(np.transpose(x_matrix), y))
     # Estimate the y coordinate of the curve data for the evaluated model parameters
    y_estimate = np.matmul(x_matrix, np.transpose(B))
    # Plot curve data, inlier data, curve estimate
    plt.scatter(x, y, label = 'Scattered data')
    plt.plot(x, y_estimate, 'r', label = 'Curve Fitting')
    plt.legend()
    
def main():
    leastSquares_curveFitting()
    
if __name__=='__main__':
    main()



# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 02:43:16 2020

@author: Arun
"""

import numpy as np
import matplotlib.pyplot as plt

def leastSquares_curveFitting():
    curve_data = np.genfromtxt('data_1.csv', delimiter = ',')
    x = curve_data[1:,0]
    y = curve_data[1:,1]
    x_squared = np.power(x,2)
    x_matrix = np.transpose([x_squared, x, np.ones(np.shape(x))])
    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_matrix), x_matrix)),
                  np.matmul(np.transpose(x_matrix), y))
    print(B)
    y_estimate = np.matmul(x_matrix, np.transpose(B))
    plt.scatter(x, y, label = 'Scattered data')
    plt.plot(x, y_estimate, 'r', label = 'Curve Fitting')
    plt.legend()
    
def main():
    leastSquares_curveFitting()
    
if __name__=='__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:30:53 2020

@author: Arun
"""


import numpy as np
import matplotlib.pyplot as plt

curve_data = np.genfromtxt('data_1.csv', delimiter = ',')
# Extract x and y coordinates from the curve data
x = curve_data[1:,0]
y = curve_data[1:,1]

x_squared = np.power(x,2)
x_matrix = np.transpose([x_squared, x, np.ones(np.shape(x))])

#Least Square Implementation
def least_square_curvefitting():
    '''
    Function to fit a curve using Least squares given minimum number of points (x,y) from the data.
    Y = XB
    E = (Y - XB)T(Y - XB)
    dE/dB = 0
    B = inv(XTX)*XT*Y 
    '''
    
    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_matrix), x_matrix)),
                  np.matmul(np.transpose(x_matrix), y))
    y_estimate = np.matmul(x_matrix, np.transpose(B))
    plt.figure()
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Data_1')
    plt.scatter(x, y, label = 'Scattered data')
    plt.plot(x, y_estimate, 'r', label = 'Curve Fitting')
    plt.legend()
    plt.savefig('least_square_curvefitting_data_1.png')
    
#Least Square with Regularization
def least_square_regularization():
    '''
    Function to fit a curve using Least squares given minimum number of points (x,y) from the data.
    Y = XB
    E = (Y - XB)T(Y - XB)
    dE/dB = 0
    B = (inv(XTX)+L)*(XT*Y)
    '''
    L=np.dot(np.identity(3),5)
    Breg = np.matmul(np.linalg.inv((np.matmul(np.transpose(x_matrix), x_matrix))+L), np.matmul(np.transpose(x_matrix), y))
    y_estimatereg = np.matmul(x_matrix, np.transpose(Breg))
    plt.figure()
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Data_1')
    plt.scatter(x, y, label = 'Scattered data')
    plt.plot(x, y_estimatereg , 'r', label = 'Curve Fitting')
    plt.legend()
    plt.savefig('least_square_regularization_data_1.png')
    
#Total Least Square
def total_least_square():
    yn=np.reshape(y,(y.shape[0],1))
    a=np.concatenate((x_matrix,yn),axis=1)
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    V = vh.T
    Vxy = V[:3,3]
    Vyy = V[3,3]
    a_tls = - Vxy / Vyy
    VV=V[:,3].reshape(4,1)
    Xtyt=np.matmul(np.matmul(a,VV),VV.T)
    Xtyt=Xtyt[:,0:3]
    y_tls=np.matmul((Xtyt+x_matrix),a_tls)
    plt.figure()
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Data_1')
    plt.scatter(x, y, label = 'Scattered data')
    plt.plot(x, y+y_tls , 'r', label = 'Curve Fitting')
    plt.legend()
    plt.savefig('total_least_square_data_1.png')

def main():
    least_square_curvefitting()
    least_square_regularization()
    total_least_square()
    
if __name__=='__main__':
    main()

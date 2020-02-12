# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:52:31 2020

@author: Praveen
"""

import numpy as np
from scipy.sparse.linalg import eigs


def calculate_SandV(A):
    '''
    Calculate right singular vectors V and obtain homography matrix H
    '''
    A_Transpose_A = np.matmul(np.transpose(A), A)
    eigen_values, eigen_vectors = eigs(A_Transpose_A, 8) 
    idx = eigen_values.argsort()[::-1]
    sorted_eigen_values = np.real(np.round(eigen_values[idx]))
    sorted_eigen_vectors = np.real(eigen_vectors[:,idx])
    # print(sorted_eigen_vectors)
    S_matrix = np.diag(np.sqrt(sorted_eigen_values))
    V = sorted_eigen_vectors
    H = np.dot(np.reshape(V[:,8],(3,3)),-1)
    # print(H)
    return S_matrix, V, eigen_values, H

def calculateU(A, V_matrix, eigen_values):
    '''
    Calculate diagonal matrix S and left singular vectors U
    '''
    '''
    A_A_Transpose = np.matmul(A, np.transpose(A))
    eigen_values, eigen_vectors = eigs(A_A_Transpose, 7)
    idx = eigen_values.argsort()[::-1]
    # sorted_eigen_values = eigen_values[idx]
    sorted_eigen_vectors = eigen_vectors[:,idx]
    U_matrix = sorted_eigen_vectors
    '''
    U_matrix = np.matmul(A, V_matrix)
    for i in range(U_matrix.shape[0]):
        U_matrix[:,i] = U_matrix[:,i]/np.sqrt(eigen_values[i])
    return U_matrix

    
def main():
    x1, y1 = 5, 5
    xp1, yp1 = 100, 100
    x2, y2 = 150, 5
    xp2, yp2 = 200, 80
    x3, y3 = 150, 150
    xp3, yp3 = 220, 80
    x4, y4 = 5, 150
    xp4, yp4 = 100, 200
    A = np.array([[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],
          [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],
          [-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],
          [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],
          [-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],
          [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],
          [-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],
          [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]], dtype = np.float64)
    
    S_matrix, V_matrix, eigen_values, H = calculate_SandV(A)
    U_matrix = calculateU(A, V_matrix, eigen_values)
    # print(S_matrix.shape)
    # print(V_matrix.shape)
    # print(U_matrix.shape)
    A_estimate = np.matmul(U_matrix, np.matmul(S_matrix, np.transpose(V_matrix)))
    print(np.round(A_estimate))
if __name__ == '__main__':
    main()
    

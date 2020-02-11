# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:30:53 2020

@author: Arun
"""


import numpy as np
import matplotlib.pyplot as plt
curve_data = np.genfromtxt('data_1.csv', delimiter = ',')
x = curve_data[1:,0]
y = curve_data[1:,1]
#Least Square Implementation
x_squared = np.power(x,2)
x_matrix = np.transpose([x_squared, x, np.ones(np.shape(x))])
B = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_matrix), x_matrix)),
                  np.matmul(np.transpose(x_matrix), y))
# print(B)
y_estimate = np.matmul(x_matrix, np.transpose(B))
plt.scatter(x, y, label = 'Scattered data')
plt.plot(x, y_estimate, 'r', label = 'Curve Fitting')
plt.legend()
plt.show()
#Least Square with Regularization
L=np.dot(np.identity(3),5)
Breg = np.matmul(np.linalg.inv((np.matmul(np.transpose(x_matrix), x_matrix))+L), np.matmul(np.transpose(x_matrix), y))
y_estimatereg = np.matmul(x_matrix, np.transpose(Breg))
plt.scatter(x, y, label = 'Scattered data')
plt.plot(x, y_estimatereg , 'r', label = 'Curve Fitting')
plt.legend()
plt.show()
#Total Least Square
yn=np.reshape(y,(y.shape[0],1))
a=np.concatenate((x_matrix,yn),axis=1)
u, s, vh = np.linalg.svd(a, full_matrices=True)
# a_tls = - V(1:n,1+n) / V(n+1,n+1)
V = vh.T
Vxy = V[:3,3]
Vyy = V[3,3]
a_tls = - Vxy / Vyy
VV=V[:,3].reshape(4,1)
Xtyt=np.matmul(np.matmul(a,VV),VV.T)
Xtyt=Xtyt[:,0:3]
y_tls=np.matmul((Xtyt+x_matrix),a_tls)
plt.scatter(x, y, label = 'Scattered data')
plt.plot(x, y+y_tls , 'r', label = 'Curve Fitting')
plt.legend()
plt.show()
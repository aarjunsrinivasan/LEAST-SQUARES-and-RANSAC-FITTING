# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy import linalg 
x1,y1=5,5
xp1,yp1=100,100
x2,y2=150,5
xp2,yp2=200,80
x3,y3=150,150
xp3,yp3=220,80
x4,y4=5,150
xp4,yp4=100,200
A=np.array([[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],
          [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],
          [-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],
          [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],
          [-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],
          [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],
          [-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],
          [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]])
# A is 8*9

RA=np.dot(A.transpose(),A)#right singular vectors 9*9(V)
LA=np.dot(A,A.transpose())#leftt singular vectors 8*8(U)
w1, v1 = linalg.eig(RA)
#Sorting the eig values and vectors in descending order


w2, v2 = linalg.eig(LA)

H=np.reshape(v1[:,8],(3,3))#Homography matrix
# A=UDV'
#A is 8*9
# U is 8*8 right singular vectors
# D is 8*9 diagonal matrix 
# V is 9*9 left singular vectors
U=v2
#V=v1.transpose()
V=v1
D=np.diag(np.sqrt(w2))
D=np.concatenate((D,np.zeros((8,1))), axis=1)
a=np.dot(np.dot(U,D),np.transpose(V))
b=np.dot(a,-1)
c=np.ceil(b)
#print(a)
#print(b)
print(c)
print(A)


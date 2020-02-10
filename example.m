clc;
clear all;

x1=5
x2=150
x3=150
x4=5
y1=5
y2=5
y3=150
y4=150
xp1=100
xp2=200
xp3=220
xp4=100
yp1=100
yp2=80
yp3=80
yp4=200

A = [-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1;0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1;-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2;0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2;-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3;0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3;-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4;0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]
transpose(A)
AAT= A*(transpose(A))

[V_AAT,D] = eig(AAT)
D = sqrt(D)
D_zeros = [D zeros(size(A,1),1)]

ATA = transpose(A)*A
[V_ATA,D] = eig(ATA)

V_AAT*D_zeros*V_ATA
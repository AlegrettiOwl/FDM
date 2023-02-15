import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import imageio
import os
import random as rnd
import time
import matplotlib.colors as colors
import matplotlib.lines as lines
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

N = 12-1
delta_r=1.9/(N)
delta_z=12/(N)
rad = np.arange(0.1,2+delta_r, step=delta_r)
r,z = np.meshgrid(rad,np.arange(-6,6+delta_z, step=delta_z))
f = np.zeros_like(r)
df = np.zeros_like(r)
fp = np.array([np.zeros_like(rad),np.zeros_like(rad)])
F = np.zeros_like(r)
S = np.zeros_like(r)
def der(f, x_i):
    derf = np.zeros_like(f)
    if x_i == 'r':
        for k in range(len(f)):
            for i in range(len(f[0])):
                if i>1 and i<len(f[0])-2:
                    derf[k][i] = (f[k][i-2]-8*f[k][i-1]+8*f[k][i+1]-f[k][i+2])/(12*delta_r)
                elif i==1 or i==len(f[0])-2:
                    if i==1:
                        a = 1
                    else:
                        a = -1
                    derf[k][i] = a*(-3*f[k][i-1*a]-10*f[k][i]+18*f[k][i+1*a]-6*f[k][i+2*a]+f[k][i+3*a])/(12*delta_r)
                else:
                    if i==0:
                        a = 1
                    else:
                        a = -1
                    derf[k][i] = a*(-25*f[k][i]+48*f[k][i+1*a]-36*f[k][i+2*a]+16*f[k][i+3*a]-3*f[k][i+4*a])/(12*delta_r)
    elif x_i == 'z':
        for k in range(len(f)):
            for i in range(len(f[0])):
                if k>1 and k<len(f)-2:
                    derf[k][i] = (f[k-2][i]-8*f[k-1][i]+8*f[k+1][i]-f[k+2][i])/(12*delta_z)
                elif k==1 or k==len(f)-2:
                    if k==1:
                        a = 1
                    else:
                        a = -1
                    derf[k][i] = a*(-3*f[k-1*a][i]-10*f[k][i]+18*f[k+1*a][i]-6*f[k+2*a][i]+f[k+3*a][i])/(12*delta_z)
                else:
                    if k==0:
                        a = 1
                    else:
                        a = -1
                    derf[k][i] = a*(-25*f[k][i]+48*f[k+1*a][i]-36*f[k+2*a][i]+16*f[k+3*a][i]-3*f[k+4*a][i])/(12*delta_z)
    else:
        print('ERROR')
        print('Por favor introduzca una direccion valida')
        input()
        exit()


    return derf

def der2(f, x_i):
    derf = np.zeros_like(f)
    if x_i == 'r':
        derf[0:, 2:-2]= (f[0:, 0:-4]-8*f[0:,1:-3]+8*f[0:,3:-1]-f[0:,4:])
        derf[0:, 1]   = (-3*f[0:,0]-10*f[0:,1]+18*f[0:,2]-6*f[0:,3]+f[0:,4])
        derf[0:, -2]  =-(-3*f[0:,-1]-10*f[0:,-2]+18*f[0:,-3]-6*f[0:,-4]+f[0:,-5])
        derf[0:, 0]   = (-25*f[0:,0]+48*f[0:,1]-36*f[0:,2]+16*f[0:,3]-3*f[0:,4])
        derf[0:, -1]  =-(-25*f[0:,-1]+48*f[0:,-2]-36*f[0:,-3]+16*f[0:,-4]-3*f[0:,-5])
        derf /= 12*delta_r
    elif x_i == 'z':
        derf[2:-2,0:]= (f[0:-4,0:]-8*f[1:-3,0:]+8*f[3:-1,0:]-f[4:,0:])
        derf[1,0:]   = (-3*f[0,0:]-10*f[1,0:]+18*f[2,0:]-6*f[3,0:]+f[4,0:])
        derf[-2,0:]  =-(-3*f[-1,0:]-10*f[-2,0:]+18*f[-3,0:]-6*f[-4,0:]+f[-5,0:])
        derf[0,0:]   = (-25*f[0,0:]+48*f[1,0:]-36*f[2,0:]+16*f[3,0:]-3*f[4,0:])
        derf[-1,0:]  =-(-25*f[-1,0:]+48*f[-2,0:]-36*f[-3,0:]+16*f[-4,0:]-3*f[-5,0:])
        derf /= 12*delta_z
    else:
        print('ERROR')
        print('Por favor introduzca una direccion valida')
        input()
        exit()


    return derf

def Lap(f):  # Solucion a la ecuacion de Poisson [nabla^2 f = V_0]
    LS = np.zeros_like(f)
    F = np.zeros_like(np.arange(len(f)*len(f[0])*1.00))
    L = np.zeros_like(np.arange(len(f)*len(f[0])*1.00))
    A = np.zeros_like(np.identity(len(f)*len(f[0])))

    for k in range(len(f)):             # Transfiriendo condiciones iniciales
        for i in range(len(f[0])):      # a vector n x m
            F[i+k*len(f[0])]=f[k][i]

    for k in range(len(f)):
        for i in range(len(f[0])):

            if k>1 and k<len(f)-2:
                A[i+k*len(f[0])][(i)+(k-2)*len(f[0])] += -1/delta_z**2
                A[i+k*len(f[0])][(i)+(k-1)*len(f[0])] += 16/delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]     += -30/delta_z**2
                A[i+k*len(f[0])][(i)+(k+1)*len(f[0])] += 16/delta_z**2
                A[i+k*len(f[0])][(i)+(k+2)*len(f[0])] += -1/delta_z**2

            elif k == 1 or k == len(f)-2:
                if k ==1:
                    b=  1
                else:
                    b= -1

                A[i+k*len(f[0])][(i)+(k-b*1)*len(f[0])] += 11/delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]       += -20/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*1)*len(f[0])] += 6/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*2)*len(f[0])] += 4/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*3)*len(f[0])] += -1/delta_z**2

            elif k == 0 or k == len(f)-1:
                if k ==0:
                    b=  1
                else:
                    b= -1

                A[i+k*len(f[0])][(i)+k*len(f[0])]       += 35/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*1)*len(f[0])] += -104/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*2)*len(f[0])] += 114/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*3)*len(f[0])] += -56/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*4)*len(f[0])] += 11/delta_z**2

            if i>1 and i<len(f[0])-2:
                A[i+k*len(f[0])][(i-2)+k*len(f[0])] += -1/delta_r**2 + 1/(delta_r*rad[i])
                A[i+k*len(f[0])][(i-1)+k*len(f[0])] += 16/delta_r**2 - 8/(delta_r*rad[i])
                A[i+k*len(f[0])][(i)+k*len(f[0])]   += -30/delta_r**2 - 0/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+1)+k*len(f[0])] += 16/delta_r**2 + 8/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+2)+k*len(f[0])] += -1/delta_r**2 - 1/(delta_r*rad[i])

            elif i == 1 or i == len(f[0])-2:
                if i ==1:
                    a=  1
                else:
                    a= -1

                A[i+k*len(f[0])][(i-a*1)+k*len(f[0])] += 11/delta_r**2 - 3*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i)+k*len(f[0])]     += -20/delta_r**2 - 10*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*1)+k*len(f[0])] += 6/delta_r**2 + 18*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*2)+k*len(f[0])] += 4/delta_r**2 - 6*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*3)+k*len(f[0])] += -1/delta_r**2 +1*a/(delta_r*rad[i])

            elif i== 0 or i == len(f[0])-1 :
                if i ==0:
                    a=  1
                else:
                    a= -1

                A[i+k*len(f[0])][(i)+k*len(f[0])]    += 35/delta_r**2 - 25*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*1)+k*len(f[0])]+= -104/delta_r**2 + 48*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*2)+k*len(f[0])]+= 114/delta_r**2 - 36*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*3)+k*len(f[0])]+= -56/delta_r**2 +16*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*4)+k*len(f[0])]+= 11/delta_r**2 - 3*a/(delta_r*rad[i])
    A = A/12
    print(np.linalg.det(A))
    F = A.dot(F)     # Solucion al sistema de ecuaciones
    for k in range(len(f)):             # Transferencia del vector solucion
        for i in range(len(f[0])):      # a la matriz de coordenadas
            LS[k][i]=F[i+len(f[0])*k]

    del A,F
    return LS

def LapS(f, fp):  # Solucion a la ecuacion de Poisson [(d^2/dr^2 + 1/r d/dr + d^2/dz^2) f = V_0]
    LS = np.zeros_like(f)
    F = np.zeros_like(np.arange(len(f)*len(f[0])*1.00))
    L = np.zeros_like(np.arange(len(f)*len(f[0])*1.00))
    A = np.zeros_like(np.identity(len(f)*len(f[0])))

    for k in range(len(f)):             # Transfiriendo condiciones iniciales
        for i in range(len(f[0])):      # a vector n x m
            F[i+k*len(f[0])]=f[k][i]
    adz = np.array([-25,48,-36,16,-3])
    adzz= np.array([-1,16,-30,16,-1])
    for k in range(len(f)):
        for i in range(len(f[0])):

            if k>1 and k<len(f)-2:
                A[i+k*len(f[0])][(i)+(k-2)*len(f[0])] = -1/delta_z**2
                A[i+k*len(f[0])][(i)+(k-1)*len(f[0])] = 16/delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]     = -30/delta_z**2
                A[i+k*len(f[0])][(i)+(k+1)*len(f[0])] = 16/delta_z**2
                A[i+k*len(f[0])][(i)+(k+2)*len(f[0])] = -1/delta_z**2
            elif k == 1 or k == len(f)-2:
                if k ==1:
                    b=  1
                else:
                    b= -1

                A[i+k*len(f[0])][(i)+(k-b*1)*len(f[0])] = 11/delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]       = -20/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*1)*len(f[0])] = 6/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*2)*len(f[0])] = 4/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*3)*len(f[0])] = -1/delta_z**2

            elif k ==0 or k==len(f)-1:
                b = 11/adz[4]
                if k ==0:
                    a=1
                    F[i+k*len(f[0])] += -b*a*fp[0][i]/delta_z
                else:
                    a=-1
                    F[i+k*len(f[0])] += -b*a*fp[-1][i]/delta_z
                A[i+k*len(f[0])][(i)+k*len(f[0])]       = (35-b*adz[0])/delta_z**2
                A[i+k*len(f[0])][(i)+(k+1*a)*len(f[0])] = (-104-b*adz[1])/delta_z**2
                A[i+k*len(f[0])][(i)+(k+2*a)*len(f[0])] = (114-b*adz[2])/delta_z**2
                A[i+k*len(f[0])][(i)+(k+3*a)*len(f[0])] = (-56-b*adz[3])/delta_z**2
                if i==0 or i ==len(f[0])-1:
                    if k ==0:
                        F[i+k*len(f[0])] = -b*a*fp[0][i]/delta_z+11*f[k][i]/(12*delta_z**2)
                    else:
                        F[i+k*len(f[0])] = -b*a*fp[-1][i]/delta_z+11*f[k][i]/(12*delta_z**2)
                    A[i+k*len(f[0])][(i)+k*len(f[0])]  += 11/delta_z**2

            if i>1 and i<len(f[0])-2:
                A[i+k*len(f[0])][(i-2)+k*len(f[0])] += -1/delta_r**2 + 1/(delta_r*rad[i])
                A[i+k*len(f[0])][(i-1)+k*len(f[0])] += 16/delta_r**2 - 8/(delta_r*rad[i])
                A[i+k*len(f[0])][(i)+k*len(f[0])]   += -30/delta_r**2- 0/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+1)+k*len(f[0])] += 16/delta_r**2 + 8/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+2)+k*len(f[0])] += -1/delta_r**2 - 1/(delta_r*rad[i])
            elif i == 1 or i == len(f[0])-2:
                if i ==1:
                    a=  1
                else:
                    a= -1
                A[i+k*len(f[0])][(i-a*1)+k*len(f[0])] += 11/delta_r**2 - 3*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i)+k*len(f[0])]     += -20/delta_r**2 - 10*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*1)+k*len(f[0])] += 6/delta_r**2 + 18*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*2)+k*len(f[0])] += 4/delta_r**2 - 6*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*3)+k*len(f[0])] += -1/delta_r**2 +1*a/(delta_r*rad[i])
            elif i== 0 or i == len(f[0])-1:
                if i ==0:
                    a=  1
                else:
                    a= -1
                if k!= 0 and k!=len(f)-1:
                    A[i+k*len(f[0])] = np.zeros_like(A[i+k*len(f[0])])
                    A[i+k*len(f[0])][i+k*len(f[0])] = 12



    A = A/12
    print(np.linalg.det(np.linalg.inv(A)))
    F = np.linalg.inv(A).dot(F)     # Solucion al sistema de ecuaciones
    for k in range(len(f)):             # Transferencia del vector solucion
        for i in range(len(f[0])):      # a la matriz de coordenadas
            LS[k][i]=F[i+len(f[0])*k]

    del A,F
    return LS

def LapS2(f):  # Solucion a la ecuacion de Poisson [nabla^2 f = V_0]
    LS = np.zeros_like(f)
    F = np.zeros_like(np.arange(len(f)*len(f[0])*1.00))
    L = np.zeros_like(np.arange(len(f)*len(f[0])*1.00))
    A = np.zeros_like(np.identity(len(f)*len(f[0])))

    for k in range(len(f)):             # Transfiriendo condiciones iniciales
        for i in range(len(f[0])):      # a vector n x m
            F[i+k*len(f[0])]=f[k][i]

    for k in range(len(f)):
        for i in range(len(f[0])):

            if k>1 and k<len(f)-2:
                A[i+k*len(f[0])][(i)+(k-2)*len(f[0])] += -1/delta_z**2
                A[i+k*len(f[0])][(i)+(k-1)*len(f[0])] += 16/delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]     += -30/delta_z**2
                A[i+k*len(f[0])][(i)+(k+1)*len(f[0])] += 16/delta_z**2
                A[i+k*len(f[0])][(i)+(k+2)*len(f[0])] += -1/delta_z**2

            elif k == 1 or k == len(f)-2:
                if k ==1:
                    b=  1
                else:
                    b= -1

                A[i+k*len(f[0])][(i)+(k-b*1)*len(f[0])] += 11/delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]       += -20/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*1)*len(f[0])] += 6/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*2)*len(f[0])] += 4/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*3)*len(f[0])] += -1/delta_z**2

            elif k == 0 or k == len(f)-1:
                if k ==0:
                    b=  1
                else:
                    b= -1

                A[i+k*len(f[0])][(i)+k*len(f[0])]       += 35/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*1)*len(f[0])] += -104/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*2)*len(f[0])] += 114/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*3)*len(f[0])] += -56/delta_z**2
                A[i+k*len(f[0])][(i)+(k+b*4)*len(f[0])] += 11/delta_z**2

            if i>1 and i<len(f[0])-2:
                A[i+k*len(f[0])][(i-2)+k*len(f[0])] += -1/delta_r**2 + 1/(delta_r*rad[i])
                A[i+k*len(f[0])][(i-1)+k*len(f[0])] += 16/delta_r**2 - 8/(delta_r*rad[i])
                A[i+k*len(f[0])][(i)+k*len(f[0])]   += -30/delta_r**2 - 0/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+1)+k*len(f[0])] += 16/delta_r**2 + 8/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+2)+k*len(f[0])] += -1/delta_r**2 - 1/(delta_r*rad[i])

            elif i == 1 or i == len(f[0])-2:
                if i ==1:
                    a=  1
                else:
                    a= -1

                A[i+k*len(f[0])][(i-a*1)+k*len(f[0])] += 11/delta_r**2 - 3*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i)+k*len(f[0])]     += -20/delta_r**2 - 10*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*1)+k*len(f[0])] += 6/delta_r**2 + 18*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*2)+k*len(f[0])] += 4/delta_r**2 - 6*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*3)+k*len(f[0])] += -1/delta_r**2 +1*a/(delta_r*rad[i])

            if i== 0 or i == len(f[0])-1:
                if i ==0:
                    a=  1
                else:
                    a= -1
                A[i+k*len(f[0])] = np.zeros_like(A[i+k*len(f[0])])
                A[i+k*len(f[0])][i+k*len(f[0])] = 12


    A = A/12

    F = np.linalg.solve(A,F)     # Solucion al sistema de ecuaciones
    for k in range(len(f)):             # Transferencia del vector solucion
        for i in range(len(f[0])):      # a la matriz de coordenadas
            LS[k][i]=F[i+len(f[0])*k]

    del A,F
    return LS

#f[0:, 0:] = np.cos(r[0:, 0:]**3)
#fp[0, 0:] = 0
#fp[-1, 0:] = 0
#S[0:, 0:] = -3*r[0:, 0:]**2*np.sin(r[0:, 0:]**3)
#S[0:, 0:] = 4*z[0:, 0:]**2+2*r[0:, 0:]**2
#f[0:, 0:] = r[0:, 0:]**2*z[0:, 0:]
#fp[0, 0:] = r[0, 0:]**2
#fp[-1, 0:] = r[-1, 0:]**2
#S[0:, 0:] = 4*z[0:, 0:]
#f[0:, 0:] = np.cos(np.pi*r[0:, 0:])*np.sin(z[0:, 0:])
#fp[0, 0:] = np.cos(np.pi*r[0, 0:])*np.cos(z[0, 0:])
#fp[-1,0:] = np.cos(np.pi*r[-1, 0:])*np.cos(z[-1, 0:])
#S[0:, 0:] =-2*f[0:, 0:]-2*np.sin(r[0:, 0:])*np.cos(z[0:, 0:])/r[0:, 0:]

#f[0:, 0:] = 2*np.sin(np.pi*r)*(z**2-6)
#fp[0, 0:] = 4*np.sin(np.pi*r[0])*z[0]
#fp[-1,0:] = 4*np.sin(np.pi*r[-1])*z[-1]

f = r**2*z**2
fp[0] = 2*z[0]*r[0]**2
fp[-1] = 2*z[-1]*r[-1]**2


df = Lap(f)
F  = df
F2 = df
F[0:, 0]= f[0:, 0]
F[0:, -1]= f[0:, -1]

F  = LapS(F,fp)
F2 = LapS2(F)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(r,z,F, color='black')
ax.plot_surface(r,z,f)
plt.show()


#plt.plot(z[0:, len(F[0])-1] ,f[0:, len(F[0])-1], 'b--')
#plt.scatter(z[0:, len(F[0])-1] ,F[0:, len(F[0])-1])
plt.plot(r[0] ,F[0]-f[0], 'r--')
plt.show()
print(F[0]-f[0])

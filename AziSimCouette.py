######################### Librerias ################################

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


####################### Datos Experimentales ########################

R_1prime= 0.70
R_2prime= 1.05
Rprime = R_2prime-R_1prime
w_1prime = 1.01*0.5   #+0.5, b0=0.0
w_2prime = 0*0.165 #+0.165
mu = 1.5		#Viscosidad Dinamica
rho= 1261		#Densidad
nu = mu/rho		#Viscosidad Cinetica

Re_e = (R_2prime-R_1prime)*R_2prime*w_2prime/nu
Re_i = (R_2prime-R_1prime)*R_1prime*w_1prime/nu
print('Numero de Reynolds del sistema Re_e: {:.2f}'.format(Re_e))
print('Numero de Reynolds del sistema Re_i: {:.2f}'.format(Re_i))

################# Datos para el sistema semejante ###################

R_2 = 1.0
R_1 = R_1prime*R_2/R_2prime
R = R_2-R_1
w_2 = w_2prime*R_2prime*Rprime/(R_2*R)
w_1 = w_1prime*R_1prime*Rprime/(R_1*R)
v_1 = R_1*w_1
v_2 = R_2*w_2
a1 = 0
a2 = 0
longitud =10*R

a = (w_1*R_1**2-w_2*R_2**2)/(R_1**2-R_2**2)     ## Constantes de la solucion
b = (1-w_2/w_1)*w_1*R_1**2/(1-R_1**2/R_2**2)          ## exacta del flujo estable

print('Re_e de la simulacion: {:.2f}'.format((R_2-R_1)*R_2*w_2/nu))
print('Re_i de Reynolds de la simulacion: {:.2f}'.format((R_2-R_1)*R_1*w_1/nu))

############### Sistema de referencias discreto ######################

N = 40                                   # Numero de celdas por direccion
delta_r = R/(N-1)                        # Dimension de la celda en r^
delta_z = longitud/(N-1+20)                 # Dimension de la celda en z^
delta_t = 10*(delta_r*delta_z)**2/(8*nu*(4*delta_r**2+4*delta_z**2)) #Intervalos de tiempo
I = 90 #int(duracion/delta_t)            #Numero de Iteraciones
nP = 10
adz = np.array([-25,48,-36,16,-3])
adzz= np.array([-1,16,-30,16,-1])

c_L = 1/(12*delta_r**2*delta_z**2)       # Constante de aproximacion de derivadas de rango 5 del operador de Laplace

rad= np.arange(R_1, R_2+delta_r/2, delta_r)                 ## Bases en cada
azm = np.arange(0, 2*np.pi+2*np.pi/(2*N-2),2*np.pi/(N-1))   ## Direccion
Z = np.arange(-longitud/2, longitud/2+delta_z/2, delta_z)   ## r,azimut, z respectivamente

#z, th, r = np.meshgrid(Z, azm, rad)  ## Volumenes de control a lo largo del sistema

print('Celdas por unidad en cada direccion~')
print('rCU={:.2f}cel/m, phiCU={:.2f}cel/rad, zCU={:.2f}cel/m'.format(1/delta_r,N/2*np.pi,1/delta_z))

rmapZ, thmapZ = np.meshgrid(rad, azm)      ## Sistema en un plano perpendicular a
rmapTh, zmapTh = np.meshgrid(rad, Z)       ## z y azimut respectivamente

SS = np.zeros_like(rad) # Solucion estacionaria a un angulo cualquiera
frames = []             # Memoria de cuadros para confeccion de GIF

###################### Funciones de aproximacion #########################

def der(f, x_i):
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

def V_solution(v_1,v_2): # Solucion para la parte estacionaria V_0 [V_0''+ V_0'-V_0/r**2 = 0]
    SS[0]=v_1
    SS[-1]=v_2
    A = np.identity(len(rad))
    for n in range(1,len(A[0])-1):
        if n>1 and n<len(A[0])-2:
            A[n][n+2] = -(delta_r/rad[n]+1)
            A[n][n+1] = 8*(2+delta_r/rad[n])
            A[n][n] = -(12*delta_r**2/rad[n]**2+30)
            A[n][n-1] = 8*(2-delta_r/rad[n])
            A[n][n-2] = (delta_r/rad[n]-1)
        else:
            if n==1:
                a = 1
            else:
                a= -1
            A[n][n+a*3] = (a*delta_r/rad[n]-1)
            A[n][n+a*2] = (4-6*a*delta_r/rad[n])
            A[n][n+a*1] = (18*a*delta_r/rad[n]+6)
            A[n][n] = -(delta_r**2/rad[n]**2+a*10*delta_r/rad[n]+20)
            A[n][n-a*1] = (11-a*3*delta_r/rad[n])

    return np.linalg.solve(A,SS)

def P_solution(V):  # Solucion a la presion para la parte estacionaria [p' = rho*V**2/r]
    p = np.zeros_like(V)
    A = np.identity(len(V[0]))
    S = np.zeros_like(rad)

    for i in range(len(V[0])):
        S[i] = 12*(delta_r**2)*rho*V[0][i]**2/rad[i]
        if i>0 and i<len(V[0])-1:
            if i<len(V[0])//2:
                a = 1
            else:
                a = -1
            A[i][i-1*a] = -3*a
            A[i][i]     = -10*a
            A[i][i+1*a] = 18*a
            A[i][i+2*a] = -6*a
            A[i][i+3*a] = 1*a

        else:
            if i==0:
                a = 1
            else:
                a = -1
            A[i][i]     = -25*a
            A[i][i+1*a] = 48*a
            A[i][i+2*a] = -36*a
            A[i][i+3*a] = 16*a
            A[i][i+4*a] = -3*a

    S = np.linalg.solve(A,S)

    for k in range(len(V)):
        for i in range(len(V[0])):
            p[k][i] = S[i]

    return p#+5437836477.4

def Lap(f,x_i):  # Solucion a la ecuacion de Poisson [nabla^2 f = V_0]
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
                if x_i == 'z':
                    c = 11/adz[4]
                    A[i+k*len(f[0])][(i)+k*len(f[0])]       = (35-c*adz[0])/delta_z**2
                    A[i+k*len(f[0])][(i)+(k+1*b)*len(f[0])] = (-104-c*adz[1])/delta_z**2
                    A[i+k*len(f[0])][(i)+(k+2*b)*len(f[0])] = (114-c*adz[2])/delta_z**2
                    A[i+k*len(f[0])][(i)+(k+3*b)*len(f[0])] = (-56-c*adz[3])/delta_z**2
                else:
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
                #if x_i == 'r':
                A[i+k*len(f[0])] = np.zeros_like(A[i+k*len(f[0])])
                A[i+k*len(f[0])][(i)+k*len(f[0])]     = 12
                #else:
                    #A[i+k*len(f[0])][(i)+k*len(f[0])]    += 35/delta_r**2 - 25*a/(delta_r*rad[i])
                    #A[i+k*len(f[0])][(i+a*1)+k*len(f[0])]+= -104/delta_r**2 + 48*a/(delta_r*rad[i])
                    #A[i+k*len(f[0])][(i+a*2)+k*len(f[0])]+= 114/delta_r**2 - 36*a/(delta_r*rad[i])
                    #A[i+k*len(f[0])][(i+a*3)+k*len(f[0])]+= -56/delta_r**2 +16*a/(delta_r*rad[i])
                    #A[i+k*len(f[0])][(i+a*4)+k*len(f[0])]+= 11/delta_r**2 - 3*a/(delta_r*rad[i])

    A = A/12
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
    F = np.linalg.inv(A).dot(F)     # Solucion al sistema de ecuaciones
    for k in range(len(f)):             # Transferencia del vector solucion
        for i in range(len(f[0])):      # a la matriz de coordenadas
            LS[k][i]=F[i+len(f[0])*k]

    del A,F
    return LS

def pert_imp(f, V0, v_phi,v_phi_n, p, x_i): # Solucion en n+1/2 a las perturbaciones de V
    x_i = x_i.lower()

    V = np.zeros_like(f)
    En = np.zeros_like(np.arange(len(f)*len(f[0])*1.00))
    A = np.zeros_like(np.identity(len(f)*len(f[0])))
    L = np.zeros_like(f)
    COR = np.zeros_like(f)

    L = nu*delta_t*Lap(f)

    if x_i == 'r':
        derP = der(p, 'r')
        for k in range(len(f)):
            for i in range(len(f[0])):
                COR[k][i]       = (2 + nu*delta_t/(rad[i]**2))
                IZ       = (2 - nu*delta_t/(rad[i]**2))*f[k][i]
                IZ      += (2*V_0[k][i]*v_phi_n[k][i]/rad[i]-derP[k][i]/rho)*delta_t
                En[i+k*len(f[0])] = L[k][i]+IZ
    elif x_i == 'phi':
        for k in range(len(f)):
            for i in range(len(f[0])):
                IZ       = (2 - nu*delta_t/(rad[i]**2))*f[k][i]
                En[i+k*len(f[0])] = L[k][i]+IZ
                COR[k][i]        = (2 + nu*delta_t/(rad[i]**2))/c_L
    elif x_i == 'z':
        derP = der(p, 'z')
        for k in range(len(f)):
            for i in range(len(f[0])):
                IZ       = 2*f[k][i]
                IZ      -= delta_t*derP[k][i]/rho
                En[i+k*len(f[0])] = L[k][i]+IZ
                COR[k][i]        = 2
    else:
        print('ERROR')
        print('Por favor introduzca una direccion valida')
        input()
        exit()

    COR = -COR/(nu*delta_t*c_L)

    for k in range(len(p)):
        for i in range(len(f[0])):

            if i>1 and i<len(f[0])-2:

                A[i+k*len(f[0])][(i-2)+k*len(f[0])] = (-1 + delta_r/rad[i])*delta_z**2
                A[i+k*len(f[0])][(i-1)+k*len(f[0])] = (16 - 8*delta_r/rad[i])*delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]   = (-30 - 0*delta_r/rad[i] )*delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]  += COR[k][i]
                A[i+k*len(f[0])][(i+1)+k*len(f[0])] = (16 + 8*delta_r/rad[i])*delta_z**2
                A[i+k*len(f[0])][(i+2)+k*len(f[0])] = (-1 - delta_r/rad[i])*delta_z**2

            elif i == 1 or i == len(f[0])-2:
                if i ==1:
                    a=  1
                else:
                    a= -1

                A[i+k*len(f[0])][(i-a*1)+k*len(f[0])] = (11 - 3*a*delta_r/rad[i])*delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]     = (-20 - 10*a*delta_r/rad[i])*delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]      += COR[k][i]
                A[i+k*len(f[0])][(i+a*1)+k*len(f[0])] = (6 + 18*a*delta_r/rad[i])*delta_z**2
                A[i+k*len(f[0])][(i+a*2)+k*len(f[0])] = (4 - 6*a*delta_r/rad[i])*delta_z**2
                A[i+k*len(f[0])][(i+a*3)+k*len(f[0])] = (-1 +1*a*delta_r/rad[i])*delta_z**2

            elif i== 0 or i == len(f[0])-1 :

                if i ==0:
                    a=  1
                else:
                    a= -1
                #A[i+k*len(p[0])][(i)+k*len(p[0])]      += (35 - 25*a*delta_r/rad[i])*delta_z**2
                #A[i+k*len(p[0])][(i+a*1)+(k)*len(p[0])] = (-104 + 48*a*delta_r/rad[i])*delta_z**2
                #A[i+k*len(p[0])][(i+a*2)+(k)*len(p[0])] = (114 - 36*a*delta_r/rad[i])*delta_z**2
                #A[i+k*len(p[0])][(i+a*3)+(k)*len(p[0])] = (-56 + 16*a*delta_r/rad[i])*delta_z**2
                #A[i+k*len(p[0])][(i+a*4)+(k)*len(p[0])] = (11 - 3*a*delta_r/rad[i])*delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]      += (35)*delta_z**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]      += COR[k][i]
                A[i+k*len(f[0])][(i+a*1)+(k)*len(f[0])] = (-104)*delta_z**2
                A[i+k*len(f[0])][(i+a*2)+(k)*len(f[0])] = (114)*delta_z**2
                A[i+k*len(f[0])][(i+a*3)+(k)*len(f[0])] = (-56)*delta_z**2
                A[i+k*len(f[0])][(i+a*4)+(k)*len(f[0])] = (11 )*delta_z**2
            if k>1 and k<len(f)-2:

                A[i+k*len(f[0])][(i)+(k-2)*len(f[0])] = -1*delta_r**2
                A[i+k*len(f[0])][(i)+(k-1)*len(f[0])] = 16*delta_r**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]    += -30*delta_r**2
                A[i+k*len(f[0])][(i)+(k+1)*len(f[0])] = 16*delta_r**2
                A[i+k*len(f[0])][(i)+(k+2)*len(f[0])] = -1*delta_r**2

            elif k == 1 or k == len(f)-2:
                if k ==1:
                    b=  1
                else:
                    b= -1

                A[i+k*len(f[0])][(i)+(k-b*1)*len(f[0])] = 11*delta_r**2
                A[i+k*len(f[0])][(i)+k*len(f[0])]      += -20*delta_r**2
                A[i+k*len(f[0])][(i)+(k+b*1)*len(f[0])] = 6*delta_r**2
                A[i+k*len(f[0])][(i)+(k+b*2)*len(f[0])] = 4*delta_r**2
                A[i+k*len(f[0])][(i)+(k+b*3)*len(f[0])] = -1*delta_r**2

            elif k == 0 or k == len(f)-1:
                if k ==0:
                    b=  1
                else:
                    b= -1

                A[i+k*len(f[0])][(i)+k*len(f[0])]      += 35*delta_r**2
                A[i+k*len(f[0])][(i)+(k+b*1)*len(f[0])] = -104*delta_r**2
                A[i+k*len(f[0])][(i)+(k+b*2)*len(f[0])] = 114*delta_r**2
                A[i+k*len(f[0])][(i)+(k+b*3)*len(f[0])] = -56*delta_r**2
                A[i+k*len(f[0])][(i)+(k+b*4)*len(f[0])] = -11*delta_r**2

    A = -nu*delta_t*c_L*A
    En = np.linalg.inv(A).dot(En)

    for k in range(len(f)):
        for i in range(len(f[0])):
            V[k][i]=En[i+len(f[0])*k]

    del En,L,A,COR

    if x_i == 'r':
        V = V-2*delta_t*V_0*v_phi
    return V

def pert_exp(V0, v_phi, v_r, p, x_i): # Solucion en n+1 a las perturbaciones
    x_i = x_i.lower()
    COR = np.zeros_like(V0)
    derP = np.zeros_like(V0)
    ex = np.zeros_like(V0)
    L = np.zeros_like(V0)

    if x_i == 'phi':
        COR[0:, 1:-1] = (1 - nu*delta_t/rad[1:-1]**2)*v_phi[0:, 1:-1]
        ex[0:, 1:-1] = -delta_t*(a-b/rad[1:-1]**2)*v_r[0:, 1:-1]
        L = nu*delta_t*Lap(v_phi,x_i)
        L[0:, -1] = 0
        L[0:, 0]  = 0
        return COR + L + ex - derP
    elif x_i == 'z':
        COR[0:, 1:-1]  = 1*v_z[0:, 1:-1]
        derP = delta_t*der(p, 'z')/rho
        L = nu*delta_t*Lap(v_z,x_i)
        derP[0:, -1] = 0
        derP[0:, 0]  = 0
        return COR + L - derP
    elif x_i == 'r':
        COR[0:, 1:-1] = (1 - nu*delta_t/rad[1:-1]**2)*v_r[0:, 1:-1]
        ex[0:, 1:-1] = 2*delta_t*V_0[0:, 1:-1]*v_phi[0:, 1:-1]/rad[1:-1]
        derP = delta_t*der(p, 'r')/rho
        #plt.pcolormesh(rmapTh, zmapTh, derP, shading = 'gouraud')
        #plt.plot(rmapTh, zmapTh, color='k', ls='none')
        L = nu*delta_t*Lap(v_r,x_i)
        derP[0:, -1] = 0
        derP[0:, 0]  = 0
        return COR + L + ex - derP
    else:
        print('ERROR')
        print('Por favor introduzca una direccion valida')
        input()
        exit()

    return 0

def pert_P(f, v_r, V_0, v_phi, v_z):
    P = np.zeros_like(f)
    der_r = np.zeros_like(f)
    der_z = np.zeros_like(f)
    F = np.zeros_like(np.arange(len(f)*len(f[0])*1.00))
    A = np.zeros_like(np.identity(len(f)*len(f[0])))
    PN = np.zeros_like(F)
    A = np.identity(len(f)*len(f[0]))

    #der_r[0:,0:] = V_0[0:,0:]*v_phi[0:,0:]+v_r[0:,0:]*rad[0:]/(2*delta_t)
    #der_r=der(der_r,'r')
    #der_r[0:,0:] = rho*der_r[0:,0:]/rad[0:]
    #der_z=rho*der(v_z , 'z')/(2*delta_t)
    der_r=2*der(v_phi*V_0,'r')
    der_r= rho*der_r[0:,0:]/rad[0:]


    for k in range(len(f)):             # Transfiriendo condiciones iniciales
        for i in range(len(f[0])):      # a vector n x m
            #F[i+k*len(f[0])]=der_r[k][i]+der_z[k][i]
            F[i+k*len(f[0])]= der_r[k][i]

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
                else:
                    a=-1
                A[i+k*len(f[0])][(i)+k*len(f[0])]       = (35-b*adz[0])/delta_z**2
                A[i+k*len(f[0])][(i)+(k+1*a)*len(f[0])] = (-104-b*adz[1])/delta_z**2
                A[i+k*len(f[0])][(i)+(k+2*a)*len(f[0])] = (114-b*adz[2])/delta_z**2
                A[i+k*len(f[0])][(i)+(k+3*a)*len(f[0])] = (-56-b*adz[3])/delta_z**2

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
                A[i+k*len(f[0])][(i)+k*len(f[0])]     = -20/delta_r**2 - 10*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*1)+k*len(f[0])] = 6/delta_r**2 + 18*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*2)+k*len(f[0])] = 4/delta_r**2 - 6*a/(delta_r*rad[i])
                A[i+k*len(f[0])][(i+a*3)+k*len(f[0])] = -1/delta_r**2 +1*a/(delta_r*rad[i])
            elif i== 0 or i == len(f[0])-1:
                if i ==0:
                    a=  1
                else:
                    a= -1
                b = 11/adz[4]
                A[i+k*len(f[0])][(i)+k*len(f[0])]       += (35-b*adz[0])/delta_r**2
                A[i+k*len(f[0])][(i+1*a)+(k)*len(f[0])] = (-104-b*adz[1])/delta_r**2
                A[i+k*len(f[0])][(i+2*a)+(k)*len(f[0])] = (114-b*adz[2])/delta_r**2
                A[i+k*len(f[0])][(i+3*a)+(k)*len(f[0])] = (-56-b*adz[3])/delta_r**2

    A = A/12
    PN+= F
    PN = np.linalg.solve(A,PN)
    #for n in range(nP):
    #    PN+= F
    #    PN = np.linalg.solve(A,PN)     # Solucion al sistema de ecuaciones
    for k in range(len(p)):             # Transferencia del vector solucion
        for i in range(len(p[0])):      # a la matriz de coordenadas
            P[k][i]=PN[i+len(p[0])*k]

    return P

########################## Inicio de simulacion ###########################

start_time= time.time()
V_0 = np.zeros_like(rmapTh)                         ## Variables de gaurdado
v_phi = np.zeros_like(V_0)                          ## para las soluciones
v_r = np.zeros_like(V_0)                            ## de las ecuaciones
v_z = np.zeros_like(V_0)                            ## diferenciales
v_phi_n = np.zeros_like(V_0)
v_r_n = np.zeros_like(V_0)
P_0 = np.zeros_like(V_0)
p = np.zeros_like(V_0)
alpha = np.zeros_like(V_0)                     # Datos de V para r constante
sigma = np.zeros_like(V_0)
gamma = np.zeros_like(V_0)
SS = V_solution(v_1,v_2)     # Solucion del campo de velocidades estacionarias

for j in range(len(V_0)):   # Transferencia de la solucion estacionaria al plano Z
    V_0[j]=SS

P_0 = P_solution(V_0)         # Solucion de la presion estacionaria

for k in range(len(v_phi)):
    for i in range(1,len(v_phi[0])-1):
        v_phi[k][i]= 0.000001*np.cos(np.pi*rad[i]/R)*np.cos(np.pi*Z[k]/longitud)
#        v_r[k][i]= (rnd.random()-0.5)*0.00001
#        v_phi[k][i]= (rnd.random()-0.5)*0.00001
#        v_z[k][i]= (rnd.random()-0.5)*0.00001
        v_z[k][i]= 0.000001*np.sin(np.pi*rad[i]/R+R_1 )*np.sin(np.pi*Z[k]/longitud)
        v_r[k][i]= 0.000001*np.sin(np.pi*Z[k]/longitud)

############################# Iteraciones ################################

for t in range(0,I):
    print("Calculando cuadro #{} de {}".format(t+1,I))
    v_1= a1*delta_t + v_1
    v_2= a2*delta_t + v_2
    v_phi_n = v_phi
    v_r_n = v_r
    p = pert_P(p, v_r_n, V_0, v_phi_n, v_z)
    ##### Derivacion Implicita
    #v_phi = pert_imp(v_phi, V_0, v_phi, v_phi_n,P_0+p, 'phi')
    #v_r = pert_imp(v_r, V_0, v_phi, v_phi_n, P_0+p, 'r')
    #v_z = pert_imp(v_z, V_0, v_phi, v_phi_n, P_0+p, 'z')

    ###### Derivacion Explicita
    v_phi = pert_exp(V_0, v_phi_n, v_r_n, p, 'phi')
    v_r = pert_exp(V_0, v_phi_n, v_r_n, p, 'r')
    v_z = pert_exp(V_0, v_phi_n, v_r_n, p, 'z')

    comp = 0
    av   = 0
    for k in range(len(V_0)):
        for i in range(len(V_0[0])):
            comp+= np.sqrt(v_r[k][i]**2+(v_phi[k][i]+V_0[k][i])**2+v_z[k][i]**2)
            av  += V_0[k][i]

#    sigma = p[N//2]                 # Datos de V para z constante
#    alpha= p[0:,len(v_z)//2]
    #gamma = der(alpha,'z')*delta_t/rho
############################### Graficos #################################
    if t%2 ==0:
        f = plt.figure()
        #ax1 = f.add_subplot(231)
        #plt.ylabel("v_r")
        #plt.plot(rad, sigma)
        #plt.setp(ax1.get_xticklabels(), visible='false')

        ax2 = f.add_subplot(222)
        plt.title("v_r")
        plt.pcolormesh(rmapTh, zmapTh, v_r, shading = 'gouraud')
        plt.plot(rmapTh, zmapTh, color='k', ls='none')

        ax4 = f.add_subplot(224)
        plt.title("v_z")
        plt.pcolormesh(rmapTh, zmapTh, v_z, shading = 'gouraud')
        plt.plot(rmapTh, zmapTh, color='k', ls='none')
        #plt.axis('off')

        #ax5 = f.add_subplot(233)
        #plt.plot(rad, V_0[len(V_0)//2]+v_phi[len(V_0)//2], rad, V_0[len(V_0)//2], 'r--')
        #plt.setp(ax1.get_xticklabels(), visible='false')

        ax6 = f.add_subplot(121)
        plt.pcolormesh(rmapTh[len(v_z)//4:-len(v_z)//4,1:-1]
                    , zmapTh[len(v_z)//4:-len(v_z)//4,1:-1]
                    , p[len(v_z)//4:-len(v_z)//4,1:-1]
                    , shading = 'gouraud')
        plt.streamplot(rmapTh[len(v_z)//4+1:-len(v_z)//4-1,2:-2]
                ,zmapTh[len(v_z)//4+1:-len(v_z)//4-1,2:-2]
                ,-v_r[len(v_z)//4+1:-len(v_z)//4-1,2:-2]
                ,v_z[len(v_z)//4+1:-len(v_z)//4-1,2:-2]
                , color = 'black', density = 1, linewidth= 0.3)

        #plt.quiver(rmapTh[len(v_z)//4:-len(v_z)//4,1:-1]
        #            , zmapTh[len(v_z)//4:-len(v_z)//4,1:-1]
        #            ,v_z[len(v_z)//4:-len(v_z)//4,1:-1]
        #            ,v_r[len(v_z)//4:-len(v_z)//4,1:-1])

        text = f.text(0.5, 0.46, 't = {:.3f}s'.format(t*delta_t), size=8)
        text = f.text(0.5, 0.42, 'v/V0 = {:.3f}'.format(comp/av), size=8)
        text = f.text(0.8, 0.42, 'Re_e = {:.2f}'.format(Re_e), size=8)
        text = f.text(0.8, 0.46, 'Re_i = {:.2f}'.format(Re_i), size=8)
        text.set_path_effects([path_effects.Normal()])
        plt.subplots_adjust(bottom=0.015, right=0.85, top=0.95, hspace=0.345)
        frame = f'{t}.png'
        frames.append(frame)
        plt.savefig(frame, dpi=200, bbox_inches='tight')
        plt.close()



############################ Calculos finales #############################

print('Tiempo promedio para el calculo de cada cuadro = {:.2f}s'.format((time.time()-start_time)/I))
print('Tiempo Total: {:.2f}min'.format((time.time()-start_time)/60))
#print('---Confeccionando animacion---')
if comp/av >= 1.05:
    St = 'D'
else:
    St = 'S'

with imageio.get_writer('FCTE(E={:.2f},I={:.2f},Res={}){}.gif'.format(Re_e,Re_i, N,St), mode='I') as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

for frame in set(frames):
    os.remove(frame)

print("ReE = {:.2f}, ReI = {:.2f}, St = {}".format(Re_e, Re_i, St))
print('Presione cualquier tecla para salir.')
input()

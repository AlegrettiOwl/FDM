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

x,y = np.meshgrid(np.arange(-3,3, step=0.1),np.arange(-3,3, step=0.1))
f = np.zeros_like(x)
df = np.zeros_like(x)
delta_r=0.1
delta_z=0.1
for i in range(len(f)):
    for k in range(len(f[0])):
        f[k][i] = 2*np.exp(-x[k][i]**2-y[k][i]**2)+0.6*np.sqrt(x[k][i]**2+y[k][i]**2)

#df = der(f,'z')
df = der(f,'r')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x,y,f)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x,y,df)
plt.show()

plt.plot(x[len(f[0])//2], f[len(f[0])//2],x[len(f[0])//2],df[len(f[0])//2],'--g')
plt.show()

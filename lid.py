######################### Librerias ################################
##Lid-Driven cavity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import imageio
import os
import random as rnd
import time
import matplotlib.colors as colors
import matplotlib.lines as lines
from matplotlib import cm, pyplot
from matplotlib.ticker import LinearLocator

############### Sistema de referencias discreto ######################
N =40                                   # Numero de celdas por direccion
dx = 1/(N-1)                        # Dimension de la celda en r^
dy = 1/(N-1)                        # Dimension de la celda en z^
dt = 0.001
#delta_t = 2*Re_e*(delta_r*delta_z)**2/(4*delta_r**2+4*delta_z**2) #Intervalos de tiempo

rho = 1
nu = 0.1
x = np.arange(0, 1+dx/2, dx)                 ## Bases en cada
y = np.arange(0, 1+dy/2, dy)

X, Y = np.meshgrid(x, y)      ## Sistema en un plano perpendicular a
u =np.zeros_like(X)
v =np.zeros_like(Y)
p =np.zeros_like(X)
U =np.zeros_like(X)
V =np.zeros_like(X)
P =np.zeros_like(X)
b =np.zeros_like(X)



nt=80
nit=10
frames = []                           # Memoria de cuadros para confeccion de GIF

#######################################################################
def Presion(p,u,v):
    pn = np.zeros_like(p)
    pf = np.zeros_like(p)
    pn = p
    alpha = np.zeros(nit)
    pf[1:-1,1:-1] = (-((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx))**2-
    ((v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))**2-
    ((u[1:-1,2:]-u[1:-1,0:-2])*(v[2:,1:-1]-v[0:-2,1:-1]))/(2*dx*dy)+
    (u[1:-1,2:]-u[1:-1,0:-2])/(2*dt*dx)+
    (v[2:,1:-1]-v[0:-2,1:-1])/(2*dt*dy))*rho*(dx*dy)**2/(dx**2+dy**2)
    for n in range(nit):
        pn = p

        p[1:-1,1:-1]=(((pn[1:-1,2:]+pn[1:-1,0:-2])*dy**2 + (pn[2:,1:-1]+pn[0:-2,1:-1])*dx**2)/(2*(dx**2+dy**2))-
                    pf[1:-1,1:-1])

        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2

    return p

def evo(u,v,p):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        p = Presion(p,u,v)


        u[1:-1, 1:-1]= (un[1:-1, 1:-1]*( 1 - dt*(un[1:-1, 1:-1]-un[1:-1, 0:-2])/dx)-
                        vn[1:-1, 1:-1]*dt*(un[1:-1, 1:-1]-un[0:-2, 1:-1])/dy-
                        dt*(p[1:-1, 2:]-p[1:-1, 0:-2])/(2*rho*dx)+
                        nu*dt*(un[1:-1, 2:]-2*un[1:-1, 1:-1]+un[1:-1, 0:-2])/dx**2+
                        nu*dt*(un[2:, 1:-1]-2*un[1:-1, 1:-1]+un[0:-2, 1:-1])/dy**2)

        v[1:-1, 1:-1]= (vn[1:-1, 1:-1]*( 1 - dt*(vn[1:-1,1:-1]-vn[0:-2,1:-1])/dy)-
                        un[1:-1,1:-1]*dt*(un[1:-1,1:-1]-un[1:-1,0:-2])/dx-
                        dt*(p[2:,1:-1]-p[0:-2,1:-1])/(2*rho*dy)+
                        nu*dt*(vn[1:-1, 2:]-2*vn[1:-1, 1:-1]+vn[1:-1, 0:-2])/dx**2+
                        nu*dt*(vn[2:, 1:-1]-2*vn[1:-1, 1:-1]+vn[0:-2, 1:-1])/dy**2)
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 2    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0

        #widht= np.sqrt(u**2 + v**2)

        fig = pyplot.figure(figsize=(11, 7), dpi=100)
        pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
        pyplot.colorbar()
        pyplot.contour(X, Y, p, cmap=cm.viridis)
        pyplot.streamplot(X[1:-1, 0:], Y[1:-1, 0:], u[1:-1,0:], v[1:-1,0:],
                            color = 'green', density = 0.65, linewidth = 0.5)
        pyplot.quiver(X, Y, u, v)
        pyplot.xlabel('X')
        pyplot.ylabel('Y');

        frame = f'{n}.png'
        frames.append(frame)
        pyplot.savefig(frame, dpi=100)
        pyplot.close()

    return u, v, p

u,v,p = evo(u,v,p)

with imageio.get_writer('Lid.gif', mode='I') as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

for frame in set(frames):
    os.remove(frame)

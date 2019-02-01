import utilities as cc
import numpy as np
import scipy as sp
from scipy import integrate
import graphing as gg
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib
import pylab as plt

#coordlist=cc.generategrid(np.array([[-10,10],[-10,10.0]]),np.array([21.0,21.0]),np.array([2.0,3.0]),'linear')
#command for running grid search routine
#coordlist=cc.gridoptimize(x0,domain,num,convergencepower=4,fatol=0.003,xtol=.01,maxiter=30,fun='rosenbrock',gridtype='linear',verbose=1)
coordlist=cc.gridoptimize([0.0,0.0], np.array([[-11.0,9.0],[-11.0,9.0]],dtype=np.float64),[15,13],convergencepower=4,fatol=0.003,xtol=.01,maxiter=30,fun='rosenbrock',gridtype='linear',verbose=1)

#Display the coordinates that have been retrieved from the optimization.
T=np.linspace(0,1,np.shape(coordlist)[0])**2
np.savetxt('coordlist.txt',coordlist,delimiter='	')
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.gray()
for i in range(np.shape(coordlist)[0]):
	ax.scatter(coordlist[i,0],coordlist[i,1],coordlist[i,2],color=(T[i],0,1-T[i]),depthshade=True,marker="+")#,color=(T[i],0,1-T[i]))
ax.set_xlim3d(-5,5)
ax.set_ylim3d(-5,5)
ax.set_zlim3d(0,20)

plotting = True
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
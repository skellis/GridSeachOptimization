from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation
import utilities as cc
import graphing as gg

def init_animation():
    global line
    line, = plt.plot(gridfull[0,0],gridfull[0,1],color="cyan",marker='o',markersize=2,linestyle='None')

def animate(i):
	j=5*np.mod(i,201)
	start=0
	stop=0
	
	if(j<lengrid):
		start=0
		stop=j
		line.set_color("cyan")
	elif(j>=lengrid and j<=lengrid+pause):
		start=lengrid
		stop=lengrid+1
		line.set_color("red")
	elif(j>=lengrid+pause and j<=2*lengrid+pause):
		start=lengrid+pause
		stop=j
		line.set_color("yellow")
	elif(j>=2*lengrid+pause and j<=2*lengrid+2*pause):
		start=2*lengrid+pause
		stop=2*lengrid+pause+1
		line.set_color("red")
	elif(j>=2*lengrid+2*pause and j<=3*lengrid+2*pause):
		start=2*lengrid+2*pause
		stop=j
		line.set_color("orange")
	elif(j>=3*lengrid+2*pause and j<=3*lengrid+3*pause):
		start=3*lengrid+2*pause
		stop=3*lengrid+2*pause+1
		line.set_color("red")
	elif(j>=3*lengrid+3*pause and j<=4*lengrid+3*pause):
		start=3*lengrid+3*pause
		stop=j
		line.set_color("purple")
	elif(j>=4*lengrid+3*pause and j<=4*lengrid+4*pause):
		start=4*lengrid+3*pause
		stop=4*lengrid+3*pause+1
		line.set_color("red")
	#print i,j,start,stop
	line.set_data(gridfull[start:stop,0],gridfull[start:stop,1])
	
	return line,

x=np.linspace(-10.0,10.0,201)
y=np.linspace(-10.0,10.0,201)
X,Y=np.meshgrid(x,y)
XY=np.stack((X.flatten(),Y.flatten())).T
Z=cc.ackley_val(XY)
Z2=Z.reshape(np.shape(X))
grid0=np.genfromtxt('gridc0.txt',delimiter='	')
grid1=np.genfromtxt('gridc1.txt',delimiter='	')
grid2=np.genfromtxt('gridc2.txt',delimiter='	')
grid3=np.genfromtxt('gridc3.txt',delimiter='	')
grid4=np.genfromtxt('gridc4.txt',delimiter='	')
grid5=np.genfromtxt('gridc5.txt',delimiter='	')
global pause
pause=55
mingrid0=grid0[grid0[:,-1].argsort()][0,:]
grid0x=np.multiply(mingrid0,np.ones((pause,3)))

mingrid1=grid1[grid1[:,-1].argsort()][0,:]
grid1x=np.multiply(mingrid1,np.ones((pause,3)))

mingrid2=grid2[grid2[:,-1].argsort()][0,:]
grid2x=np.multiply(mingrid2,np.ones((pause,3)))

mingrid3=grid3[grid3[:,-1].argsort()][0,:]
grid3x=np.multiply(mingrid3,np.ones((pause,3)))

mingrid4=grid4[grid4[:,-1].argsort()][0,:]
grid4x=np.multiply(mingrid4,np.ones((pause,3)))

mingrid5=grid5[grid5[:,-1].argsort()][0,:]
grid5x=np.multiply(mingrid5,np.ones((pause,3)))
global gridfull
gridfull=np.vstack((grid0,grid0x,grid1,grid1x,grid2,grid2x,grid3,grid3x,grid4,grid4x,))
#numframes=np.shape(gridfull)[0]/5
numframes=220
global lengrid
lengrid=np.shape(grid0)[0]
#numframes=10
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
fig=plt.figure()
plt.rcParams['animation.convert_path'] = 'C:\imagemagick\magick.exe'
im = plt.imshow(Z2, interpolation='bilinear', origin='lower', cmap=cm.jet,extent=(X[0,0]	, X[-1,-1], Y[0,0], Y[-1,-1]))
#levels = np.arange(np.amin(Z),np.amax(Z)/20,(np.amax(Z)-np.amin(Z))/400)
levels=np.linspace(0,np.amax(Z2),num=10)
CS = plt.contour(Z2, levels, origin='lower', linewidths=1, colors='w',extent=(X[0,0]	, X[-1,-1], Y[0,0], Y[-1,-1]))
zc = CS.collections
plt.setp(zc, linewidth=1)
plt.title('Grid Search Optimize Ackley')
plt.flag()
ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init_animation, frames=numframes)
if 1:
	ani.save('C:\Users\skellis\Documents\CodingStudying\InstrumentOpt11\AckleyOpt0.gif', writer='imagemagick', fps=30)
else:
	plt.show()
import numpy as np
from numpy.linalg import inv
import scipy as sp
from scipy.special import erf


def gridoptimize(x0, domain,num,convergencepower=2.0,fatol=.0001,xtol=.001,maxiter=20,fun='rosenbrock',gridtype='linear',verbose=False):
	"""
	Executes a search of consecutilvely smaller grids looking for a local optimum. Works in spaces upto five dimensions.


	Parameters
	----------
	fun: string 
		The functions to be evaluated in this optimization. probably rosenbrock_val
	x0: ndarray
		initial guess for the search especially important is gridtype=antigauspace and erfspace 
		Also determines which coordinates are search first within a grid i.e. most proximate first
		ex. x0=[-2,5]
		generate initial grid
	domain : ndarray
		array dimension 2 x N formatted as np.array([[xmin,xmax],[ymin,ymax],[zmin,zmax],....,[Nmin,Nmax]]) which contains the space
		spanned by a given grid.  is an nx2 dimensional array which stipulated the space of which we will initially be seaching
		ex. domain=[[-5,5],[-10,10],[-7,7]] for three dimensional grid
	num: ndarray
		N dimensional array The number of points in each dimensions on which the grid is formed. 
		If domain is larger in one dimesnion it is advised to use proportionally more points in that dimension.
	convergencepower: float
		after each grid is swepts this factor determines how much the domain is scaled by. 
		Greater convergencepowers will yield faster optimizations but could deviate from the true.
	fatol: float
		function evaluated tolerance determines end condition for optimization. When the subsequent grid is only decrease by less than fatol
	xtol: float
		space tolerance determines end condition for optimization. When the subsequent domain size is sufficiently small the optimization is aborted.
	maxiter: int
		number of iterations through grids is desired before the algorithm terminates.
	gridtype: string 
		choose from three options of grids linearly space gaussian spaced or error function spaced.
	verbose: boolean 
		option to print the results of the optimization as it comes in

	Returns
	-------
	coordlist: ndarray
		N+1 x (M x iterations) dimensional array where N is the dimensionality. 
		The +1 is the measured function. M is the np.multiply(num) and iterations is the number of grids that have been generated.
		ex. coordlist=[[x0,y0,z0...N0, f(x0,y0...N0)], [x1,y1,z1...N1, f(x1,y1...N1)].....[xM,yM,zM...NM, f(xM,yM,...NM)]]
		which is an ordered list of the coordinates.
	
	Examples
	--------
	>>> coordlist=cc.gridoptimize('rosenbrock',[1.2,1.0,1.3], np.array([[-10.1,10.2],[-86.3,70.4],[-1.4,10.3]],dtype=np.float64),[9,21,7],convergencepower=2.0,fatol=0.01,xtol=.05,maxiter=30,gridtype='erfspace',verbose=1)
	>>> #coordlist=cc.gridoptimize('simple',[1.2,1.0,1.3], np.array([[-10.1,10.2],[-86.3,70.4],[-1.4,10.3]],dtype=np.float64),[9,21,7],convergencepower=2.0,fatol=0.01,xtol=.05,maxiter=30,gridtype='erfspace',verbose=1)
	
	Graphical illustration:
	--------
	coordlist=cc.gridoptimize('rosenbo',[1.2,0.90], np.array([[-10.8,11.4],[-18.4,10.3]],dtype=np.float64),[13,13],convergencepower=2,fatol=0.03,xtol=.1,maxiter=30,gridtype='linear',verbose=1)
	T=np.linspace(0,1,np.shape(coordlist)[0])**2
	np.savetxt('coordlist.txt',coordlist,delimiter='	')

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	plt.gray()
	for i in range(np.shape(coordlist)[0]):
		ax.scatter(coordlist[i,0],coordlist[i,1],coordlist[i,2],color=(T[i],0,1-T[i]),depthshade=True,marker=",")#,color=(T[i],0,1-T[i]))
	ax.set_xlim3d(-2,5)
	ax.set_ylim3d(-2,5)
	ax.set_zlim3d(-1,500)

	plotting = True
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.show()
	"""
	#dimensionality of space
	dim=len(x0)
	if verbose:
		print "Dimensionality:", dim
	
	fmin=float('Inf')
	#set up empty coordlist which we will append to
	coordlist=np.array([], dtype=np.int64).reshape(0,dim+1)
	#create scaling array
	scale=np.full(dim,1/float(convergencepower))
	#Start a new grid.
	for i in range(maxiter):
		grid,gridindex=generateNDgrid(domain,num,x0,gridtype)
		#print "grid: ",grid, "gridindex: ",gridindex
		#gather the indexing array so we can reference specific grid points after sort
		e=np.vstack((grid,gridindex))
		#sort the grid points by proximity to guess x0
		og,oi=distcoordsort(e,x0)
		#evaluate the coordinate
		#choose the correct function to evaluate the coordinates.
		if fun=='parabola':
			fog=np.array([parabola_val(og)]).T
		elif fun=='ackley':
			fog=np.array([ackley_val(og)]).T
		elif fun=='rosenbrock':
			fog=np.array([rosenbrock_val(og)]).T
		else:
			fog=np.array([rosenbrock_val(og)]).T
		#gather the  values with teh coords and index
		gfog=np.hstack((og,oi,fog))
		#sort by values smallest to largest
		hfog=gfog[gfog[:,-1].argsort()]
		#create array with exclusively relevant data
		ifog=np.delete(hfog,[range(dim,2*dim)],axis=1)
		jfog=np.delete(gfog,[range(dim,2*dim)],axis=1)
		#current minimum value is at top
		ftemp=hfog[0,-1]
		minloc=hfog[0,0:dim]
		minindex=hfog[0,dim:2*dim]
		#sort out the grid with the coordinates closest to the minimum value
		x0=minloc
		#calculate the largest domain for comparison with xtol
		xtemp=np.amax(domain[:,1]-domain[:,0])
		if verbose:
			print "_________________Iteration: ", i,"___________________"
			print "Grid Domain: " ,domain
			print "Old value: ", fmin, "New Value:" ,ftemp
			print "Minimum location: ",minloc
			print "Minimum index: ",minindex
			print "Change: " ,fmin-ftemp
			np.savetxt('gridc'+str(i)+'.txt',jfog,delimiter='	')
		if np.isnan(ftemp):
			break
		#Update and Continue search conditions
		if ftemp>fmin or ftemp<fmin-fatol:
			coordlist=np.vstack((ifog,coordlist))
			fmin=ftemp
			domain=reducedomain(domain,x0=x0,scale =scale)
		#End conditions
		elif abs(fmin-ftemp)<=float(fatol) and ftemp<=fmin:
			coordlist=np.vstack((ifog,coordlist))
			break
		#instrument precision is met
		elif xtemp<float(xtol):
			coordlist=np.vstack((ifog,coordlist))
			break
	if verbose:
		print "_____________________________________________________"
		print "Total number of measurements: ", np.shape(coordlist)[0]
	return coordlist



def rosenbrock_val(coord,*params):
	"""
	evaluates rosenbrock's function in any dimension. the minimum is found at (x,y)=(a,a^2) traditionally a is set to 1 while b is set to 100.
	Parameters
	--------------
	coord : np.array(dtype=float)
		``an N dimensional numpy array to be evaluated by Rosenbrocks function.
	params : float
		``optional positional parameters of lenght 2N which are to be the coefficients of 
		Rosenbrock's value. default is a=1 b=100
	Returns
	-------
	val : float
		value of rosenbrock's function at the given coord.
	Notes
	--------
	https://en.wikipedia.org/wiki/Rosenbrock_function
	Examples
	---------
	>>> print rosenbrock_val([1,1,1])
	[0]

	"""
	coord=np.array(coord).T
	val =0
	if len(params)==2*(len(coord)-1):
		for i in range(0, len(coord)-1):
			val+= (params[2*i]-coord[i])**2+params[2*i+1]*(coord[i+1]-coord[i]**2)**2
	else:
		for i in range(0, len(coord)-1):
			val+= (1-coord[i])**2+100*(coord[i+1]-coord[i]**2)**2
	return val

def parabola_val(coord,*params):
	"""
	evaluates a parabola's function in any dimension. the minimum is found at (x,y)=(1,1) unless alternative parameters are found
	Parameters
	--------------
	coord : np.array(dtype=float)
		``an N dimensional numpy array to be evaluated.
	params : float
		``optional positional parameters of lenght N which are to be the coefficients of 
		Minimum of the parabola default is (1,1)
	Returns
	-------
	val : float
		value of rosenbrock's function at the given coord.
	Notes
	--------
	https://en.wikipedia.org/wiki/Test_functions_for_optimization
	Examples
	---------
	>>> print rosenbrock_val([1,1,1])
	[0]

	"""
	coord=np.array(coord).T
	val =0
	if len(params)==len(coord):
		for i in range(0, len(coord)):
			val+= (coord[i]-params[i])**2
	else:
		for i in range(0, len(coord)):
			val+= (coord[i]-1)**2
	return val

def ackley_val(coord,*params):
	"""
	evaluates a parabola's function in any dimension. the minimum is found at (x,y)=(1,1) unless alternative parameters are found
	Parameters
	--------------
	coord : np.array(dtype=float)
		``an N dimensional numpy array to be evaluated.
	params : float
		``optional positional parameters of lenght N which are to be the coefficients of 
		Minimum of the parabola default is (1,1)
	Returns
	-------
	val : float
		value of rosenbrock's function at the given coord.
	Notes
	--------
	https://en.wikipedia.org/wiki/Test_functions_for_optimization
	Examples
	---------
	>>> print rosenbrock_val([1,1,1])
	[0]

	"""
	coord=np.array(coord).T
	val = -20*np.exp(-0.2*(0.5*((coord[0]-1)**2+(coord[1]-1)**2))**0.5)-np.exp(0.5*(np.cos(2*3.1415259*(coord[0]-1))+np.cos(2*3.1415259*(coord[1]-1))))+2.71828+20
	return val


def erfspace(start, stop, num=50, sig=0,mu=-27.13, endpoint=True, dtype=None,fatol=.01,maxiter=70):
	"""
	Return numbers spaced evenly on a absolute sigmoidal scale.
	spacing =abs(erf((x-mu)/2**.5/sigma))

	In linear space, the sequence starts at ``base ** start``
	(`base` to the power of `start`) and ends with ``base ** stop``
	(see `endpoint` below).

	Parameters
	----------
	start : float
		``base ** start`` is the starting value of the sequence.
	stop : float
		``base ** stop`` is the final value of the sequence, unless `endpoint`
		is False.  In that case, ``num + 1`` values are spaced over the
		interval in log-space, of which all but the last (a sequence of
		length `num`) are returned.
	sig : float, optional
		standard deviation of signmoidal distribution.  Default is (stop-start)/2.
	mu : float, optional
		Centers of signmoidal distribution.  Default is (start+stop)/2.
	num : integer, optional
		Number of samples to generate.  Default is 50.
	endpoint : boolean, optional
		If true, `stop` is the last sample. Otherwise, it is not included.
		Default is True.
	dtype : dtype
		The type of the output array.  If `dtype` is not given, infer the data
		type from the other input arguments.

	Returns
	-------
samples : ndarray
		`num` samples, where the coordinate spacing go as a the absolute value of the error funnction centered around mu

	See Also
	--------
	arange : Similar to linspace, with the step size specified instead of the
			 number of samples. Note that, when used with a float endpoint, the
			 endpoint may or may not be included.
	linspace : Similar to logspace, but with the samples uniformly distributed
			   in linear space, instead of log space.
	geomspace : Similar to logspace, but with endpoints specified directly.

	Notes
	-----
	Warning Function can give degenerate values if num is small. It is recommended that you use odd values of num and values over 5 to avoid this

	Examples
	--------
	>>> print cc.erfspace(2.0, 3.0, num=4)
	[2.         2.21670852 2.43341703 3.        ]
	>>> print cc.erfspace(2.00, 3.00, num=8,endpoint=False,dtype=float)
	[2.         2.32021714 2.54448816 2.66010857 2.66010857 2.77572898 3.         3.32021714]
	>>> print cc.erfspace(2.0, 3.0,sig=.1,mu=1.5,num=4)
	[2.         2.33333333 2.66666667 3.        ]



	Graphical illustration:
	--------
	>>> import matplotlib.pyplot as plt
	>>> import matplotlib
	>>> x=np.linspace(-11,4)
	>>> y=cc.erfspace(-11,4)
	>>> matplotlib.rcParams['axes.unicode_minus'] = False
	>>> fig, ax = plt.subplots()
	>>> ax.plot(x,y,'r+')
	>>> plt.show()
	"""
	if sig==0:
		sig=(stop-start)/2
	if mu==-27.13:
		mu=(stop+start)/2
	muTemp=mu

	for i in range(maxiter):
		a = np.linspace(start, stop, num=num-1)
		b=abs(sp.special.erf(np.nan_to_num((a-muTemp)/sig/2**.5)))
		midloc=np.where(b == b.min())
		c=np.insert(b, 0, 0., axis=0)
		c=np.cumsum(c)
		#Scale the space
		c=c*(stop-start)/(c[-1]-c[0])
		#set correct sart and stop
		c=c-c[0]+start
		midcoord=c[midloc[0]+1]
		if len(midloc)==2:
			pseudomu=(c[midloc[0]+1]+c[midloc[1]+1])/2
		#elif beta[midloc[0]-1]==gamma[midloc[0]+1]:
		else:
			pseudomu=c[midloc[0]+1]
		if abs(pseudomu[0]-mu)<fatol:
			break
		muTemp=muTemp*mu/pseudomu
	if dtype is None:
		return c
	return c.astype(dtype)


def antigauspace(start, stop, num=50, sig=0,mu=-27.13, endpoint=True, dtype=None,fatol=.01,maxiter=70):
	"""
	Return numbers spaced evenly on an anti-gaussian scale.

	In linear space, the sequence starts at ``base ** start``
	(`base` to the power of `start`) and ends with ``base ** stop``
	(see `endpoint` below).

	Parameters
	----------
	start : float
		``base ** start`` is the starting value of the sequence.
	stop : float
		``base ** stop`` is the final value of the sequence, unless `endpoint`
		is False.  In that case, ``num + 1`` values are spaced over the
		interval in log-space, of which all but the last (a sequence of
		length `num`) are returned.
	sig : float, optional
		standard deviation of signmoidal distribution.  Default is (stop-start)/2.
	mu : float, optional
		Centers of signmoidal distribution.  Default is (start+stop)/2.
	num : integer, optional
		Number of samples to generate.  Default is 50.
	endpoint : boolean, optional
		If true, `stop` is the last sample. Otherwise, it is not included.
		Default is True.
	dtype : dtype
		The type of the output array.  If `dtype` is not given, infer the data
		type from the other input arguments.
	fatol : float, optional
		Absolute error in xopt between iterations that is acceptable for convergence.
	maxiter, : int
		Maximum allowed number of iterations . Will default to N*70, If both maxiter and maxfev are set, minimization will stop at the first reached.

	Returns
	-------
	samples : ndarray
		`num` samples, where the coordinate spacing goes as a the opposite of a gaussian distribution (1-g) centered around mu

	See Also
	--------
	arange : Similar to linspace, with the step size specified instead of the
			 number of samples. Note that, when used with a float endpoint, the
			 endpoint may or may not be included.
	linspace : Similar to logspace, but with the samples uniformly distributed
			   in linear space, instead of log space.
	geomspace : Similar to logspace, but with endpoints specified directly.

	Notes
	-----


	Examples
	--------
	>>> print cc.gaussianspace(2.0, 3.0, num=4)
	[2.         2.10774696 2.21549392 3.        ]
	>>> print cc.gaussianspace(2.00, 3.00, num=4,endpoint=False,dtype=float)
	[2.   2.25 2.25 2.5 ]
	>>> print cc.gaussianspace(2.0, 3.0,sig=.1,mu=1.5,num=4)
	[2.         2.33333333 2.66666667 3.        ]

	Graphical illustration:
	>>> import matplotlib.pyplot as plt
	>>> x=np.linspace(-11,4)
	>>> y=cc.gaussianspace(-11,4)
	>>> matplotlib.rcParams['axes.unicode_minus'] = False
	>>> fig, ax = plt.subplots()
	>>> ax.plot(x,y,'r+')
	>>> plt.show()

	"""
	if sig==0:
		sig=(stop-start)/2
	if mu==-27.13:
		mu=(stop+start)/2
	muTemp=mu

	for i in range(maxiter):
		a = np.linspace(start, stop, num=num-1)
		b=1-np.exp(-np.nan_to_num((a-muTemp)**2/(2*sig**2)))
		midloc=np.where(b == b.min())
		c=np.insert(b, 0, 0., axis=0)
		c=np.cumsum(c)
		c=c*(stop-start)/(c[-1]-c[0])
		c=c-c[0]+start
		midcoord=c[midloc[0]+1]
		if len(midloc)==2:
			pseudomu=(c[midloc[0]+1]+c[midloc[1]+1])/2
		#elif beta[midloc[0]-1]==gamma[midloc[0]+1]:
		else:
			pseudomu=c[midloc[0]+1]
		if abs(pseudomu[0]-mu)<fatol:
			break
		muTemp=muTemp*mu/pseudomu
	if dtype is None:
		return c
	return c.astype(dtype)



def generateNDgrid(domain,num,x0,gridtype='linear'):
	"""
	Generates a grid of coordinates the type of which is either linearly spaced antigaussian spaced 
	or spaced as the absolute value of an error function depending on gridtype. Handles up to 5 dimensions.

	"""
	if len(x0)==1:
		if gridtype=='linear':
			x=np.linspace(domain[0,0], domain[0,1],num[0])
		elif gridtype=='antigauspace':
			x=antigauspace(domain[0,0], domain[0,1],num[0],mu=x0[0])
		elif gridtype=='erfspace':
			x=erfspace(domain[0,0], domain[0,1],num[0],mu=x0[1])
		gridindex=np.arange(num[0])
		grid=x
	elif len(x0)==2:
		if gridtype=='linear':
			x=np.linspace(domain[0,0], domain[0,1],num[0])
			y=np.linspace(domain[1,0],domain[1,1],num[1])
		elif gridtype=='antigauspace':
			x=antigauspace(domain[0,0], domain[0,1],num[0],mu=x0[0])
			y=antigauspace(domain[0,0], domain[0,1],num[1],mu=x0[1])
		elif gridtype=='erfspace':
			x=erfspace(domain[0,0], domain[0,1],num[0],mu=x0[1])
			y=erfspace(domain[0,0], domain[0,1],num[1],mu=x0[1])
		X,Y=np.meshgrid(x,y)
		grid=np.stack((X.flatten(),Y.flatten()))
		xindex=np.arange(num[0])
		yindex=np.arange(num[1])
		Xindex,Yindex=np.meshgrid(xindex,yindex)
		gridindex=np.stack((Xindex.flatten(),Yindex.flatten()))
	elif len(x0)==3:
		if gridtype=='linear':
			x=np.linspace(domain[0,0], domain[0,1],num[0])
			y=np.linspace(domain[1,0],domain[1,1],num[1])
			z=np.linspace(domain[2,0],domain[2,1],num[2])
		elif gridtype=='antigauspace':
			x=antigauspace(domain[0,0], domain[0,1],num[0],mu=x0[0])
			y=antigauspace(domain[1,0], domain[1,1],num[1],mu=x0[1])
			z=antigauspace(domain[2,0], domain[2,1],num[2],mu=x0[2])
		elif gridtype=='erfspace':
			x=erfspace(domain[0,0], domain[0,1],num[0],mu=x0[0])
			y=erfspace(domain[1,0], domain[1,1],num[1],mu=x0[1])
			z=erfspace(domain[2,0], domain[2,1],num[2],mu=x0[2])
		X,Y,Z=np.meshgrid(x,y,z)
		grid=np.stack((X.flatten(),Y.flatten(),Z.flatten()))
		xindex=np.arange(num[0])
		yindex=np.arange(num[1])
		zindex=np.arange(num[2])
		Xindex,Yindex,Zindex=np.meshgrid(xindex,yindex,zindex)
		gridindex=np.stack((Xindex.flatten(),Yindex.flatten(),Zindex.flatten()))
	elif len(x0)==4:
		if gridtype=='linear':
			x=np.linspace(domain[0,0], domain[0,1],num[0])
			y=np.linspace(domain[1,0],domain[1,1],num[1])
			z=np.linspace(domain[2,0],domain[2,1],num[2])
			w=np.linspace(domain[3,0], domain[3,1],num[3])
		elif gridtype=='antigauspace':
			x=antigauspace(domain[0,0], domain[0,1],num[0],mu=x0[0])
			y=antigauspace(domain[1,0], domain[1,1],num[1],mu=x0[1])
			z=antigauspace(domain[2,0], domain[2,1],num[2],mu=x0[2])
			w=antigauspace(domain[3,0], domain[3,1],num[3],mu=x0[3])
		elif gridtype=='erfspace':
			x=erfspace(domain[0,0], domain[0,1],num[0],mu=x0[0])
			y=erfspace(domain[1,0], domain[1,1],num[1],mu=x0[1])
			z=erfspace(domain[2,0], domain[2,1],num[2],mu=x0[2])
			w=erfspace(domain[3,0], domain[3,1],num[3],mu=x0[3])
		X,Y,Z,W=np.meshgrid(x,y,z,w)
		grid=np.stack((X.flatten(),Y.flatten(),Z.flatten(),W.flatten()))
		xindex=np.arange(num[0])
		yindex=np.arange(num[1])
		zindex=np.arange(num[2])
		windex=np.arange(num[3])
		Xindex,Yindex,Zindex,Windex=np.meshgrid(xindex,yindex,zindex,windex)
		gridindex=np.stack((Xindex.flatten(),Yindex.flatten(),Zindex.flatten(),Windex.flatten()))
	elif len(x0)==5:
		if gridtype=='linear':
			x=np.linspace(domain[0,0], domain[0,1],num[0])
			y=np.linspace(domain[1,0],domain[1,1],num[1])
			z=np.linspace(domain[2,0],domain[2,1],num[2])
			w=np.linspace(domain[3,0], domain[3,1],num[3])
			v=np.linspace(domain[4,0], domain[4,1],num[4])
		elif gridtype=='antigauspace':
			x=antigauspace(domain[0,0], domain[0,1],num[0],mu=x0[0])
			y=antigauspace(domain[1,0], domain[1,1],num[1],mu=x0[1])
			z=antigauspace(domain[2,0], domain[2,1],num[2],mu=x0[2])
			w=antigauspace(domain[3,0], domain[3,1],num[3],mu=x0[3])
			v=antigauspace(domain[4,0], domain[4,1],num[4],mu=x0[4])
		elif gridtype=='erfspace':
			x=erfspace(domain[0,0], domain[0,1],num[0],mu=x0[0])
			y=erfspace(domain[1,0], domain[1,1],num[1],mu=x0[1])
			z=erfspace(domain[2,0], domain[2,1],num[2],mu=x0[2])
			w=erfspace(domain[3,0], domain[3,1],num[3],mu=x0[3])
			v=erfspace(domain[4,0], domain[4,1],num[4],mu=x0[4])
		X,Y,Z,W,V=np.meshgrid(x,y,z,w,v)
		grid=np.stack((X.flatten(),Y.flatten(),Z.flatten(),W.flatten(),V.flatten()))
		xindex=np.arange(num[0])
		yindex=np.arange(num[1])
		zindex=np.arange(num[2])
		windex=np.arange(num[3])
		vindex=np.arange(num[4])
		Xindex,Yindex,Zindex,Windex,Vindex=np.meshgrid(xindex,yindex,zindex,windex,vindex)
		gridindex=np.stack((Xindex.flatten(),Yindex.flatten(),Zindex.flatten(),Windex.flatten(),Vindex.flatten()))
	return grid,gridindex

def distancematrix(coords):
	"""
	function for calculating the Euclidiean distance matrix for coordinates of any dimensionality.

	Parameters
	----------
	start : ndarray
		N x M dimensional array of coordinates

	Returns
	-------
	dmat : ndarray
		M^N dimensional array containing the distance matrix


	Notes
	-----
	https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy

	Examples
	--------
	>>> data=np.array([np.arange(5),np.arange(5)])
	>>> print distancematrix(data)

	[[0.         1.41421356 2.82842712 4.24264069 5.65685425]
 	[1.41421356 0.         1.41421356 2.82842712 4.24264069]
 	[2.82842712 1.41421356 0.         1.41421356 2.82842712]
 	[4.24264069 2.82842712 1.41421356 0.         1.41421356]
 	[5.65685425 4.24264069 2.82842712 1.41421356 0.        ]]

	"""
	dim=np.shape(coords)[0]
	q1=coords[0,:]
	q2=coords[1,:]
	for i in range(dim-1):
		qqcom=np.array([np.vectorize(complex)(q1,q2)])
		dmat=abs(qqcom.T-qqcom)
		q1=dmat
		if i+2<dim:
			q2=coords[i+2,:]
	return dmat


def getpolycoefs1d(coordlist):
	"""

	Parameters
	----------
	coordlist : np.array([float]) of dimesnion N x M where N is an odd value and M is aribrary while greater than N.
	coordlist must be formatted as np.array[X1,X2....XN]

	Returns
	-------
	coefs : ndarray
	coefficients of a N-1 dimensional polynomial of the form f(X1,X2...XN)=coefs[0]*X1^2+coefs[1]*X2^2+coefs[2]*X1+coefs[3]*X2+coefs[4]

	See Also
	--------
	https://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points

	Examples
	--------

	Graphical illustration:
	>>>coefs2=cc.guessmin2(points)
	>>>x=np.linspace(-10,10,num=21)
	>>>y=coefs[0]*x**2+coefs[1]*x+coefs[2]
	>>>if 1:
	>>>	saveFig=0
	>>>	autoClose=0
	>>>	matplotlib.rcParams['axes.unicode_minus'] = False
	>>>	fig, ax = plt.subplots()
	>>>	ax.plot(x,y,'bo',points[:,0],points[:,1],'gx')
	>>>	if saveFig:
	>>>		plt.savefig('slice'+str(desiredFreq)+'cm.pdf')
	>>>	if autoClose:
	>>>		plt.show(block=False)
	>>>		plt.pause(3)
	>>>		plt.close()
	>>>	else:
	>>>		plt.show()

	"""
	rcoord=coordlist[0:3,:]
	coordmat=np.array([rcoord[:,0]**2,rcoord[:,0],np.ones(3)]).T
	invmat=inv(coordmat)
	coefs=np.matmul(invmat,rcoord[:,1])
	polymin=-coefs[1]/2/coefs[0]
	return coefs

def distcoordsort(coordlist,x0):
	"""
	Calculates the distance of each coordinate in assumed to be located in rows 0 to dim in coordlist from the coordinate x0. Then sorts the 
	list by increasing distance and then splits the list in two. 
	returns the sorted list and the index list. 


	"""
	dim=len(x0)
	distance=np.power([np.sum(np.power(np.subtract(coordlist[0:dim,0:].T,x0),2),axis=1)],0.5)
	cdist=np.vstack((coordlist,distance))
	ocdist=cdist[:,cdist[-1,:].argsort()]
	ocoord=np.delete(ocdist.T,-1,axis=1)
	return ocoord[:,0:dim],ocoord[:,dim:2*dim]

def reducedomain(domain,**keyword_parameters):
	"""
	Generate a new domain with a reduced size, centereed around x0 or the middle of the previous domain. 


	Parameters
	----------
	domain : ndarray
		array dimension 2 x N formatted as np.array([[xmin,xmax],[ymin,ymax],[zmin,zmax],....,[Nmin,Nmax]]) which contains the space
		spanned by a given grid. 
	**keyword_parameters['scale']: optional ndarray
		dimension N describing how each dimension of the domain should be scaled.
	**keyword_parameters['relshift']: optional ndarray
		dimension N describing how each dimension of the domain should be shifted relative to the original domain size.
	**keyword_parameters['x0']: optional ndarray
		dimension N locating the midpoint of the newly formed domain. Defaults to the middle of the previous domain. 

	Returns
	-------
	domain2 : ndarray
		array of dimensions 2 x N similar to domain which has a been shifted and scaled.

	Examples
	--------

	Graphical illustration:

	"""
	dim=np.shape(domain)[0]
	if ('scale' not in keyword_parameters):
		scale=np.ones(dim)
	else:
		scale=keyword_parameters['scale']
	if ('relshift' not in keyword_parameters):
		relshift=np.zeros(dim)
	else:
		relshift=keyword_parameters['relshift']
	if ('x0' not in keyword_parameters):
		mid=np.mean(domain,axis=0)
	else:
		mid=keyword_parameters['x0']
	dif=domain[:,1]-domain[:,0]
	domain2=np.array([]).reshape(0,2)
	for i in range(dim):
		domain2=np.vstack((domain2,np.array([mid[i]-dif[i]/2*scale[i]+dif[i]/2*relshift[i],mid[i]+dif[i]/2*scale[i]+dif[i]/2*relshift[i]])))
	return domain2


def getpolycoefs(coordlist):
	"""
	Returns the coefficients describing a polynomial which is defined by the coordinates at the top of coordlist
	Poly nomial has the form a1X1^2+a2X2^2+a3X3^2....+aNXN^2+aN+1X1+aN+2X2+aN+3X3+....a2N+3XN+a2N+1
	Parameters
	----------
	coordlist : np.array([float]) of dimesnion N x M where N is an odd value and M is aribrary while greater than N.
	coordlist must be formatted as np.array[X1,X2....XN]
	

	Returns
	-------
	coefs : ndarray
	coefficients of a N-1 dimensional polynomial of the form f(X1,X2...XN)=coefs[0]*X1^2+coefs[1]*X2^2+coefs[2]*X1+coefs[3]*X2+coefs[4]

	See Also
	--------
	https://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points

	Notes
	-----


	Examples
	--------

	Graphical illustration:
	>>>	points=np.array([[0.0,0.0,1.0],[2.0,0.0,5.0],[0.0,1.0,9.0],[-2.0,0.0,5.0],[0.0,-2.0,5.0]])
	>>>	#points=np.array([[0.0,0.0],[2.0,2.0],[-2.0,2.0]])
	>>>	coefs=cc.getpolycoords(points)
	>>>	x=np.linspace(-3,3,num=61)
	>>>	if len(coefs)==3:
	>>>		y=coefs[0]*x**2+coefs[1]*x+coefs[2]
	>>>		matplotlib.rcParams['axes.unicode_minus'] = False
	>>>		fig, ax = plt.subplots()
	>>>		ax.plot(x,y,'bo',points[:,0],points[:,1],'gx')
	>>>		plt.show()
	>>>	if len(coefs)==5:
	>>>		y=np.linspace(-3,3,num=61)
	>>>		X,Y=np.meshgrid(x,y)
	>>>		Z=coefs[0]*X**2+coefs[1]*Y**2+coefs[2]*X+coefs[3]*Y+coefs[4]
	>>>		fig = plt.figure()
	>>>		ax = fig.add_subplot(111, projection='3d')
	>>>	ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
	>>>	ax.scatter(points[:,0],points[:,1],points[:,2],'gx')
	>>>	plt.show()

	"""
	polydim=np.shape(coordlist)[1]-1
	rcoord=coordlist[0:2*polydim+1,:]
	coordmat=np.ones((2*polydim+1)**2)
	expval=np.ones(2*polydim+1)
	expval[0:polydim]=2
	expval[polydim:2*polydim]=1
	expval[-1]=0
	for i in range(2*polydim+1):
		coordmat[(2*polydim+1)*i:(2*polydim+1)*(i+1)]=rcoord[:,np.mod(i,polydim)]**expval[i]
	coordmat=coordmat.reshape(2*polydim+1,2*polydim+1).T
	invmat=inv(coordmat)
	coefs=np.matmul(invmat,rcoord[:,-1])
	return coefs

def getpolymin(coefs):
	polymin=np.array([], dtype=np.int64)
	for i in range((len(coefs)-1)/2):
		print "index:",i+(len(coefs)-1)/2, i
		mincoord=-coefs[i+(len(coefs)-1)/2]/2/coefs[i]
		polymin=np.hstack([polymin,mincoord])
	return polymin



def calc_parabola_min(x1, y1, x2, y2, x3, y3):
		'''
		Adapted and modifed to get the unknowns for defining a parabola from three points:
		http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
		'''

		denom = (x1-x2) * (x1-x3) * (x2-x3);
		A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
		B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
		C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

		return -B/2/A



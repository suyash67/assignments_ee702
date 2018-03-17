import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from skimage import filter

def pq2fg(p,q):
	f = 2*p/(1 + (1+ p**2 + q**2)**(0.5))
	g = 2*q/(1 + (1+ p**2 + q**2)**(0.5))
	return f,g

def fg2pq(f,g):
	p = 4*f/(4-f**2-g**2)
	q = 4*g/(4-f**2-g**2)
	return p,q

def depth_calc(r,imgSize):
	[x,y]= np.meshgrid(range(0,imgSize),range(0,imgSize))
	z = np.zeros((imgSize,imgSize))  
	selected = np.zeros((imgSize,imgSize))
	for i in range(0,imgSize):
		for j in range(0,imgSize):
			z[i][j]  =  r**2 - (x[i][j]-imgSize/2)**2 - (y[i][j]-imgSize/2)**2 
			if z[i][j]>0:
				selected[i][j] = 1
				#print i,j
	z = abs(np.sqrt(z*selected))
	return z,selected

def pq_calc(z, selected):
	p = np.zeros((imgSize,imgSize))
	q = np.zeros((imgSize,imgSize))
	for i in range(1,imgSize-1):
		for j in range(1,imgSize-1):
			p[i][j] = z[i][j] - z[i][j-1]
			q[i][j] = z[i][j] - z[i-1][j]
	p, q = p*selected,q*selected
	return p,q

def rad_calc(p,q,s,imgSize,selected):
	radiance = np.zeros((imgSize,imgSize))
	for i in range(0,imgSize):
		for j in range(0,imgSize):
			radiance[i][j] = (p[i][j]*s[0] + q[i][j]*s[1] + 1)/(((s[0]**2+s[1]**2 + 1)**0.5)*((p[i][j]**2 + q[i][j]**2 + 1)**0.5))
			if (radiance[i][j] < 0 ):
				radiance[i][j] = 0

	radiance = radiance *selected
	return radiance

def init_fg(imgSize, radiance):
	boundary = np.zeros((imgSize, imgSize))

	#calculate E_x and E_y to intialize
	E_x = np.array(radiance,copy = True);
	E_y = np.array(radiance,copy = True);
	E_x[:,1:-1] = 0.5*(E_x[:,2:] - E_x[:,:-2])
	E_y[1:-1,:] = 0.5*(- E_y[2:,:] + E_y[:-2,:])
	f,g = pq2fg(E_x,E_y)
	boundary = filter.canny(radiance,sigma = 3.0);
	f_init = np.array(f*boundary,copy =True)
	g_init = np.array(g*boundary,copy =True)

	return f_init,g_init,boundary

def rad2pq(radiance,boundary,f,g,s,lamda,N): ###to review
	f_new,g_new = np.array(f,copy = True),np.array(g,copy = True)
	f_est,g_est = np.array(f,copy = True),np.array(g,copy = True)

	roi = radiance > 0
	for itr in range(0,N):
		for i in range(1,imgSize-1):
			for j in range(1,imgSize-1):
				if roi[i][j] == 1:
					rad_f,rad_g = 0,0
					rad_num = 16*(s[0]*f[i][j] + s[1]*g[i][j]) + (4-f[i][j]**2-g[i][j]**2)*(4-s[0]**2-s[1]**2)
					rad_den = (4 + f[i][j]**2 + g[i][j]**2)*(s[0]**2 + s[1]**2 + 4);
					rad = rad_num / rad_den
					rad_f = (-1*(16*s[0] - 2*f[i][j]*(4-s[0]**2-s[1]**2)*rad_den) - rad_num*(2*f[i][j]*(4+s[0]**2+s[1]**2)))/rad_den**2
					rad_g = (-1*(16*s[1] - 2*g[i][j]*(4-s[0]**2-s[1]**2)*rad_den) - rad_num*(2*g[i][j]*(4+s[0]**2+s[1]**2)))/rad_den**2
					f_est[i][j] = (f_new[i-1][j] + f_new[i+1][j] + f_new[i][j-1] + f_new[i][j+1])/4 + 1/lamda * (radiance[i][j] - rad) * rad_f
					g_est[i][j] = (g_new[i-1][j] + g_new[i+1][j] + g_new[i][j-1] + g_new[i][j+1])/4 + 1/lamda * (radiance[i][j] - rad) * rad_g 

		f_new = f_est*roi*(1-boundary) + f_new*boundary*roi
		g_new = g_est*roi*(1-boundary) + g_new*boundary*roi
	p,q = fg2pq(f_new,g_new)
	return p,q,roi


def depth_recalc(p,q,roi,depthN):
	z_init = np.zeros(p.shape)
	z = np.zeros(p.shape)
	p_x = np.array(p,copy=True)
	q_y = np.array(q,copy=True)
	p_x[:][1:-1] = 0.5*( p[:][2:] - p[:][:-2])
	q_y[1:-1][:] = 0.5*( q[:-2][:] - q[2:][:])
	for itr in range(0,depthN):
		for i in range(1,z_init.shape[0]-1):
			for j in range(1,z_init.shape[0]-1):
				if roi[i][j]:
					z[i][j] = 0.25*( z_init[i-1][j] + z_init[i+1][j] + z_init[i][j-1] + z_init[i][j+1]) + abs(p_x[i][j]) + abs(q_y[i][j])

		#print roi.shape,z.shape	
		z_init = roi* z

	return z_init


r = 20
imgSize = 50
s = [0,0,1]
depth = 1000
depthN =1000
lamda = 10
z,roi =  depth_calc(r,imgSize)
[cols,rows] = np.meshgrid(range(0,imgSize),range(0,imgSize))
# r_clip = r*0.98
# roi = r_clip**2 - (cols-imgSize/2)**2 - (rows-imgSize/2)**2 >0
# z = z * roi
p,q = pq_calc(z,roi)

radiance = rad_calc(p,q,s,imgSize,roi)


p_new,q_new,boundary = init_fg(imgSize,radiance)
p_est,q_est,roiRad = rad2pq(radiance,boundary,p_new,q_new,s,lamda,depth)
z_calc = depth_recalc(p_est,q_est,roiRad,depthN)
# roiRad = radiance>0
# z_calc = depth_recalc(p,q,roiRad,depthN)


z_value = z_calc

#print(np.amax(z_value))
plt.imshow(z_value)
#plt.show()
#plt.imshow(boundary)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d(0,imgSize)
ax.set_ylim3d(0,imgSize)
ax.set_zlim3d(0,imgSize)

surf = ax.plot_surface(rows, cols, z_value, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=1, aspect=5)
plt.show()

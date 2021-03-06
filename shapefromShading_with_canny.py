# for part 0) assuming object to be sphere at z0 and s at (1,0,0). Hence we know depth and s^ vector.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from skimage import filter
# generation of sphere
# calculation of depth : z^2 = r^2 - x^2 -y^2
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

# E(x,y) = 1 + p*ps + q*qs/sqrt((1+p*p+q*q)(1+ps*ps+qs*qs)) 
# E(x,y) = 0 if n.s is -ve
def rad_calc(p,q,s,imgSize,selected):
	radiance = np.zeros((imgSize,imgSize))
	for i in range(0,imgSize):
		for j in range(0,imgSize):
			radiance[i][j] = (p[i][j]*s[0] + q[i][j]*s[1] + 1)/(((s[0]**2+s[1]**2 + 1)**0.5)*((p[i][j]**2 + q[i][j]**2 + 1)**0.5))
			if (radiance[i][j] < 0 ):
				radiance[i][j] = 0

	radiance = radiance *selected
	# im = plt.imshow(radiance,cmap = 'hot')
	# plt.colorbar(im, orientation = 'horizontal' )
	# plt.show()
	return radiance

	#im = plt.imshow(radiance,cmap = 'hot')
	#plt.colorbar(im, orientation = 'horizontal' )
	#plt.show()
	

# initialization points for p,q

def init_pq(imgSize, radiance):
	boundary = np.zeros((imgSize, imgSize))

	#calculate E_x and E_y to intialize
	E_x = np.array(radiance,copy = True);
	E_y = np.array(radiance,copy = True);
	E_x[:,1:-1] = 0.5*(E_x[:,2:] - E_x[:,:-2])
	E_y[1:-1,:] = 0.5*(- E_y[2:,:] + E_y[:-2,:])

	boundary = filter.canny(radiance,sigma = 1.0);
	p_init = np.array(E_x*boundary,copy =True)
	q_init = np.array(E_y*boundary,copy =True)

	return p_init,q_init,boundary	
	


def rad2pq(radiance,boundary,p,q,s,lamda,N):
	p_new,q_new = np.array(p,copy = True),np.array(q,copy = True)
	p_est,q_est = np.array(p,copy = True),np.array(q,copy = True)

	roi = radiance > 0
	for itr in range(0,N):
		for i in range(1,imgSize-1):
			for j in range(1,imgSize-1):
				if roi[i][j] == 1:
					rad_p,rad_q = 0,0
					rad = (p_new[i][j]*s[0] + q_new[i][j]*s[1] + 1)/(((s[0]**2+s[1]**2 + 1)**0.5)*((p_new[i][j]**2 + q_new[i][j]**2 + 1)**0.5))
					rad_p = (q_new[i][j]**2*s[0] + s[0] -p_new[i][j]*q_new[i][j]*s[1] -p_new[i][j])/(((s[0]**2 + s[1]**2 + 1)**0.5)*((p_new[i][j]**2 + q_new[i][j]**2 + 1)**1.5))
					rad_q = (p_new[i][j]**2*s[1] + s[1] -p_new[i][j]*q_new[i][j]*s[0] -q_new[i][j])/(((s[0]**2 + s[1]**2 + 1)**0.5)*((p_new[i][j]**2 + q_new[i][j]**2 + 1)**1.5))
					p_est[i][j] = (p_new[i-1][j] + p_new[i+1][j] + p_new[i][j-1] + p_new[i][j+1])/4 + 1/lamda * (radiance[i][j] - rad) * rad_p
					q_est[i][j] = (q_new[i-1][j] + q_new[i+1][j] + q_new[i][j-1] + q_new[i][j+1])/4 + 1/lamda * (radiance[i][j] - rad) * rad_q 
					
		p_new = p_est*roi*(1-boundary) + p_new*boundary*roi
		q_new = q_est*roi*(1-boundary) + q_new*boundary*roi
	return p_new,q_new,roi


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
r_clip = r*0.98
roi = r_clip**2 - (cols-imgSize/2)**2 - (rows-imgSize/2)**2 >0
z = z * roi
p,q = pq_calc(z,roi)

radiance = rad_calc(p,q,s,imgSize,roi)



p_new,q_new,boundary = init_pq(imgSize,radiance)
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
#e = np.sum(abs(z - z_calc)**2)
#print e 



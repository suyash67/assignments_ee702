# for part 0) assuming object to be sphere at z0 and source at (1,0,0). Hence we know depth and s^ vector.

import numpy as np
#import pylab as plt
#import cv2
import matplotlib.pyplot as plt

from skimage import measure
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
	z = np.sqrt(z*selected)
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
	#print radiance[45][45]
	radiance = radiance *selected
	return radiance

	#im = plt.imshow(radiance,cmap = 'hot')
	#plt.colorbar(im, orientation = 'horizontal' )
	#plt.show()
	#im = cv2.imread(radiance)
	#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	#ret,thresh = cv2.threshold(imgray,127,255,0)
	#contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(im,contours,-1,(0,255,0),3)

# initialization points for p,q
def init_pq(imgSize,radiance):
	boundary = np.zeros((imgSize,imgSize))
	roi  = radiance > 0
	contours = measure.find_contours(roi, 0.0)
	a,b = contours[0][2]
	print roi[int(a-1)][int(b-1)]

	'''
	# Display the image and plot all contours found
	fig, ax = plt.subplots()
	ax.imshow(roi, interpolation='nearest', cmap=plt.cm.gray)

	for n,contour in enumerate(contours):
	    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

	ax.axis('image')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()
	'''


def rad2pq(radiance,p,q,s,roi,lamda,N):
	p_new,q_new = np.array(p,copy = True),np.array(q,copy = True)
	for itr in range(0,N):
		for i in range(1,imgSize-1):
			for j in range(1,imgSize-1):
				if roi[i][j] == 1:
					rad_p,rad_q = 0,0
					rad = (p[i][j]*s[0] + q[i][j]*s[1] + 1)/(((s[0]**2+s[1]**2 + 1)**0.5)*((p[i][j]**2 + q[i][j]**2 + 1)**0.5))
					rad_p = (q[i][j]**2*s[0] + s[0] -p[i][j]*q[i][j]*s[1] -p[i][j])/(((s[0]**2 + s[1]**2 + 1)**0.5)*((p[i][j]**2 + q[i][j]**2 + 1)**1.5))
					rad_q = (p[i][j]**2*s[1] + s[1] -p[i][j]*q[i][j]*s[0] -q[i][j])/(((s[0]**2 + s[1]**2 + 1)**0.5)*((p[i][j]**2 + q[i][j]**2 + 1)**1.5))
					p_new[i][j] = (p[i-1][j] + p[i+1][j] + p[i][j-1] + p[i][j+1])/4 + 1/lamda * (radiance[i][j] - rad) * rad_p
					p_new[i][j] = (q[i-1][j] + q[i+1][j] + q[i][j-1] + q[i][j+1])/4 + 1/lamda * (radiance[i][j] - rad) * rad_q 

	# if on boundary then don't update otherwise updated version


#def depth_recalc(p,q):
	

r = 25
imgSize = 100
z,roi =  depth_calc(r,imgSize)
s = [0,0,1]

p,q = pq_calc(z,roi)

radiance = rad_calc(p,q,s,imgSize,roi)
init_pq(imgSize,radiance)


import numpy as np
# import pylab as plt
# import cv2
import matplotlib.pyplot as plt
from skimage import measure
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as ndi
from skimage import feature,morphology,io
import PIL.Image
from functions import rad2pq, depth_recalc
from skimage.transform import resize, rescale, rotate, setup
from Init_PQ import init_pq


# def image2pq(img,src,lambda,noiseRad,noiseSrc,nDepth):
#     # img1 =
#
#
#     return

img1 = 'buddhaStatue.jpg';
img = io.imread(img1, as_grey=True);
imgSize = 128;
img = resize(img,(imgSize,imgSize))
radiance = img;

depth = 10000
depthN = 10000
lamda = 1000
s = [0, 0, 1]


p_new, q_new, boundary = init_pq(imgSize, radiance,canny=True);
p_est,q_est,roiRad = rad2pq(radiance,boundary,p_new,q_new,s,lamda,depth)
z_calc = depth_recalc(p_est,q_est,roiRad,depthN)





from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

z_value = z_calc
plt.imshow(z_value)
# plt.show()fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d(0, imgSize)
ax.set_ylim3d(0, imgSize)
ax.set_zlim3d(0, imgSize)
[cols, rows] = np.meshgrid(range(0, imgSize), range(0, imgSize))
surf = ax.plot_surface(rows, cols, z_value, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=1, aspect=5)
plt.show()












#
# imgCol = img.shape[0]; imgRow = img.shape[1];
#
# boundary = feature.canny(img);
#
#
# maxsize = (128, 128)
# tn_image = img.resize([256,512],PIL.Image.ANTIALIAS)
#
# img.resize((500,400),Image.NEAREST)
#
#
#
#
# plt.imshow(z_calc)
# plt.show()
#
#
#

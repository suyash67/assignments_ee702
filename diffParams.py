import numpy as np
# import pylab as plt
# import cv2
import matplotlib.pyplot as plt
from skimage import measure
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as ndi
from skimage import feature,morphology,io
import PIL.Image
from functions import rad2pq, depth_recalc,depth_calc,pq_calc,rad_calc
# from ShapeFromShading import depth_calc,pq_calc
from skimage.transform import resize, rescale, rotate, setup
from Init_PQ import init_pq
from random import uniform
import matplotlib.image as mpimg
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



base_dir = "/home/ankit/Desktop/CV/ShapeFromShading/"
saveDir2D = os.path.join(base_dir,"Sphere2D")
saveDir3D = os.path.join(base_dir,"Sphere3D")

lamdal = [1,10,100,1000,10000];
noiseAmpl = [0,0.1,0.2,0.35,0.5];              #source noise
radAmpl = [0,0.1,0.2,0.5,1];

# radAmp = 0; noiseAmp = 0; lamda = 10;


for lamda in lamdal:
    for noiseAmp in noiseAmpl:
        for radAmp in radAmpl:

            print("Lambda = {:2f} SourceNoise = {:2f} and RadianceNoise = {:2f}".format(lamda,noiseAmp,radAmp));
            r = 10
            imgSize = 50
            z, roi = depth_calc(r, imgSize)
            s = list(np.array([0, 0, 1]) + noiseAmp * np.array([uniform(0,1),uniform(0,1),uniform(0,1)]))
            depth = 1000
            depthN = 1000


            p, q = pq_calc(z, roi);
            radiance = rad_calc(p, q, s, imgSize, roi)

            #add Noise in radiance
            # radAmplitude  = np.amax(radiance);
            radiance = radiance + radAmp * np.random.normal(0,1,size= [radiance.shape[0],radiance.shape[1]]);

            p_new, q_new, boundary = init_pq(imgSize, radiance)
            p_est, q_est, roiRad = rad2pq(radiance, boundary, p_new, q_new, s, lamda, depth)
            # z_calc = depth_recalc(p_est,q_est,roiRad,depthN)

            roiRad = radiance > 0
            z_calc = depth_recalc(p, q, roiRad, depthN)

            #------------------------------------------------------------------------ Plot Graphs
            # generate file name and save image
            figName2D = "SpherePQ_2d_lambda_" + str(lamda) + "_NoiseAmp_" + str(noiseAmp) + "_RadAmp_" + str(radAmp) + ".png";
            ImagePath2D = os.path.join(saveDir2D,figName2D);
            mpimg.imsave(ImagePath2D,z_calc);

            # generate 3d sphere and save file
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_xlim3d(0, imgSize)
            ax.set_ylim3d(0, imgSize)
            ax.set_zlim3d(0, imgSize)
            [cols, rows] = np.meshgrid(range(0, imgSize), range(0, imgSize))
            surf = ax.plot_surface(rows, cols, z_calc, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=1, aspect=5)


            # generate file name and save image
            figName3D = "SpherePQ_3d_lambda_" + str(lamda) + "_NoiseAmp_" + str(noiseAmp) + "_RadAmp_" + str(radAmp) + ".png";
            ImagePath3D = os.path.join(saveDir3D,figName3D);

            plt.savefig(ImagePath3D)
            plt.close(fig)




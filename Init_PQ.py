import numpy as np
# import pylab as plt
# import cv2
import matplotlib.pyplot as plt

# from skimage import measure
from skimage import filter
from skimage.morphology import erosion, dilation



#Initializes the pq values to suitable values using boundary
def init_pq(imgSize, radiance,canny = False):
    boundary = np.zeros((imgSize, imgSize))

    # calculate E_x and E_y to intialize
    E_x = np.array(radiance, copy=True);
    E_y = np.array(radiance, copy=True);
    E_x[:, 1:-1] = 0.5 * (E_x[:, 2:] - E_x[:, :-2])
    E_y[1:-1, :] = 0.5 * (E_y[2:, :] - E_y[:-2, :])
    if canny:
        boundary = filter.canny(radiance);
    # Replacing with Erosion and Dilation
    else:
        roiRad = radiance > 0;
        kernel = np.ones([3, 3]);
        boundary = roiRad - erosion(roiRad, kernel);
        boundary = erosion(dilation(boundary, kernel), kernel);

    p_init = E_x * boundary;
    q_init = E_y * boundary;


    return p_init, q_init, boundary


#------------------------------------------------------------- Test Codes
# img1 = 'buddhaStatue.jpg';
# img = io.imread(img1, as_grey=True);
# imgSize = 128;
# img = resize(img,(imgSize,imgSize))
# radiance = img;


# p_new, q_new, boundary = init_pq(imgSize, radiance)






# print  p_init
# # # Display the image and plot all contours found
# # fig, ax = plt.subplots()
# #im = plt.imshow(p_init,cmap = 'hot')
# # plt.colorbar(im, orientation = 'horizontal' )
# plt.show()
# ax.imshow(p_init, interpolation='nearest', cmap=plt.cm.gray)
# for n,contour in enumerate(contours):
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()
#

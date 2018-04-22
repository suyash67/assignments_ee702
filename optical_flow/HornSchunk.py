import os
import imageio
from compareGraphs import compareGraphs
from hornschunck import HornSchunck
from translation import opt2translation,opt2rotation
import scipy


# base_dir = '/home/smokersan/Desktop/Optical-Flow-master/OpticalFlow'

# img0 = os.path.join(base_dir,'box_0.bmp')
# img1 = os.path.join(base_dir,'box_1.bmp')
# img0 = imageio.imread(img0, as_gray=True)
# img1 = imageio.imread(img1, as_gray=True)
# lamda = 1.;
# n = 100;

# u,v = HornSchunck(img0, img1, 1., 100)
# compareGraphs(u,v, img1,scale=9, fn="Test")



vid_path = '/home/smokersan/Desktop/Optical-Flow-master/OpticalFlow/dhakkan.mp4'
reader = imageio.get_reader(vid_path)

for i, im in enumerate(reader):
    if i < (reader._get_length() - 1) and i%20== 0 and i<40:
        img0 = reader.get_data(i);
        img1 = reader.get_data(i+45); # at least i + 25 rakho

        # img0 = scipy.misc.imresize(img0[:,:,0],0.3)
        # img1 = scipy.misc.imresize(img1[:,:,0],0.3)
        img0 = scipy.ndimage.gaussian_filter(img0,sigma = 1.0)
        img1 = scipy.ndimage.gaussian_filter(img1,sigma = 1.0)
        # img0 = scipy.ndimage.median_filter(img0, 3)
        # img1 = scipy.ndimage.median_filter(img1, 3)
        

        img0 = scipy.misc.imresize(img0[:,:,0],(200,200))
        img1 = scipy.misc.imresize(img1[:,:,0],(200,200))
        img0 = scipy.ndimage.rotate(img0,270,reshape = False)
        img1 = scipy.ndimage.rotate(img1,270,reshape = False)
        

        #compute u,v parameters
        u, v = HornSchunck(img0, img1, 200, 100) # if i + 25 is increased then increase alpha then optical flow will decrease
        compareGraphs(u,v, img1,scale=3,quivstep=7, fn="Image{:1}".format(i))
        opt2translation(u,v)


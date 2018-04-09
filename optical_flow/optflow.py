import cv2
import numpy as np
from matplotlib import pyplot as plt

def optFlow(img1,img2,lamda,n):
    #set up initial velocities
    uInit = np.zeros([img1.shape[0],img1.shape[1]])
    vInit = np.zeros([img1.shape[0],img1.shape[1]])

    # Set initial value for the flow vectors
    u = uInit
    v = vInit


    # Averaging kernel
    # kernel=np.matrix([[1/12, 1/6, 1/12],[1/6, 0, 1/6],[1/12, 1/6, 1/12]])
    kernel=np.matrix([[0, 1/4, 0],[1/4, 0, 1/4],[0, 1/4, 0]])

    # Estimate derivatives
    Ex, Ey, Et = derivatives(img1, img2)

    # Iteration to reduce error
    for i in range(n):

        # Compute averages of the flow vectors
        uAvg = cv2.filter2D(u,-1,kernel)
        vAvg = cv2.filter2D(v,-1,kernel)

        uNumer = (Ex.dot(uAvg) + Ey.dot(vAvg) + Et).dot(Ex)
        uDenom = 1 + lamda*(Ex**2 + Ey**2)
        u = uAvg - np.divide(uNumer,uDenom)

        # print np.linalg.norm(u)

        vNumer = (Ex.dot(uAvg) + Ey.dot(vAvg) + Et).dot(Ey)
        vDenom = 1 + lamda*(Ex**2 + Ey**2)
        v = vAvg - np.divide(vNumer,vDenom)
    return (u,v)

def derivatives(img1,img2):
    Ex,Ey,Et = np.array(img1,copy=True),np.array(img1,copy=True),np.array(img1,copy=True)
    for i in range(1,img1.shape[0]-1):
        for j in range(1,img1.shape[0]-1):
            Ex[i][j] = (0.25*img1[i+1][j] + 0.25*img1[i+1][j+1] + 0.25*img2[i+1][j] + 0.25*img2[i+1][j+1]) - (0.25*img1[i][j] + 0.25*img1[i][j+1] + 0.25*img2[i][j]+ 0.25*img2[i][j+1])
            Ey[i][j] = (0.25*img1[i][j+1] + 0.25*img1[i+1][j+1] + 0.25*img2[i][j+1] + 0.25*img2[i+1][j+1]) - (0.25*img1[i][j] + 0.25*img1[i+1][j] + 0.25*img2[i][j]+ 0.25*img2[i+1][j])
            Et[i][j] = (0.25*img2[i][j] + 0.25*img2[i+1][j+1] + 0.25*img2[i+1][j] + 0.25*img2[i][j+1]) - (0.25*img1[i][j] + 0.25*img1[i][j+1] + 0.25*img1[i+1][j]+ 0.25*img1[i+1][j+1])
    return Ex,Ey,Et

def smoothImage(img,kernel):
    G = gaussFilter(kernel)
    smoothedImage=cv2.filter2D(img,-1,G)
    smoothedImage=cv2.filter2D(smoothedImage,-1,G.T)
    return smoothedImage

def gaussFilter(segma):
    kSize = 2*(segma*3)
    x = range(-kSize/2,kSize/2,1+1/kSize)
    x = np.array(x)
    G = (1/(2*np.pi)**.5*segma) * np.exp(-x**2/(2*segma**2))
    return G

def compareGraphs():
    # plt.ion() #makes it so plots don't block code execution
    plt.imshow(imgNew,cmap = 'gray')
    # plt.scatter(POI[:,0,1],POI[:,0,0])
    for i in range(len(u)):
        if i%5 ==0:
            for j in range(len(u)):
                if j%5 == 0:
                    plt.arrow(j,i,v[i,j]*.0001,u[i,j]*.0001, color = 'red')
                pass
        # print i
    # plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)
    plt.show()

def opt2rotation(u,v):
    a,b,c,d,e,f,k,l,m =0,0,0,0,0,0,0,0,0
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            a = a + ((i+1)*(i+1)*(j+1)*(j+1) + ((j+1)*(j+1)+1)^2)
            b = b + ((i+1)*(i+1)*(j+1)*(j+1) + ((i+1)*(i+1)+1)^2)
            c = c + ((j+1)^2 + (i+1)^2)
            d = d - ((i+1)*(j+1)*((i+1)^2 + (j+1)^2 +2))
            e = e - (j+1)
            f = f - (i+1)
            k = k + (u[i][j]*(i+1)*(j+1) + v[i][j]*((j+1)^2 + 1))
            l = l - (u[i][j]*((i+1)^2 + 1) + v[i][j]*(i+1)*(j+1))
            m = m + (u[i][j]*(j+1) - v[i][j]*(i+1))
    M = np.matrix([[a, d, f],[d, b, e],[f, e, c]])
    n = np.matrix([[k],[l],[m]])
    # print M.shape , n.shape
    w = np.matmul(np.linalg.inv(M),n)
    print w
count = 0

#upload images# 
directory = 'box/box.'
# directory = 'office/office.'
# directory = 'rubic/rubic.'
# directory = 'sphere/sphere.'
fileName = directory + str(count) + '.bmp'
imgOld = cv2.imread(fileName,0)
# imgOld = cv2.GaussianBlur(imgOld,(FILTER,FILTER),1)
imgOld = smoothImage(imgOld,1)

count += 1
imgNew = cv2.imread(fileName,0)
# imgNew = cv2.GaussianBlur(imgNew,(FILTER,FILTER),1)
imgNew = smoothImage(imgNew,1)



[u,v] = optFlow(imgOld, imgNew, 1, 100)
# print u.shape
compareGraphs()
opt2rotation(u,v)
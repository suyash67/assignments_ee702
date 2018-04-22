import numpy as np
import scipy

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
    # print w

def opt2translation(u,v):
    a,b,c,d,e,f,alpha,beta =0,0,0,0,0,0,0,0
    depth = np.zeros(u.shape)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            a = a + (v[i][j])*(v[i][j])
            b = b + (u[i][j])*(u[i][j])
            c = c + ((i+1)*v[i][j] - (j+1)*u[i][j])*((i+1)*v[i][j] - (j+1)*u[i][j])
            d = d - (u[i][j]*v[i][j])
            e = e + (u[i][j]*((i+1)*v[i][j] - (j+1)*u[i][j]))
            f = f - (u[i][j]*((i+1)*v[i][j] - (j+1)*u[i][j]))
    G = np.matrix([[a, d, f],[d, b, e],[f, e, c]])
    eigen,vect = np.linalg.eig(G)
    arg_e1 = np.argmin(eigen)
    # print(eigen,eigen[arg_e1])
    U,V,W = vect[:,arg_e1]/np.amin(np.absolute(vect[:,arg_e1]))
    U,V,W = U*100,V*100,W*100
    # U,V,W = U*(norm[0][0])^0.5,V*(norm[0][0])^0.5,W*(norm[0][0])^0.5
    print (U,V,W)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            alpha = -U + (i+1)*W
            beta = -V + (j+1)*W
            # depth1[i][j] = alpha/u[i][j]
            depth[i][j] = (alpha* alpha + beta*beta)/(u[i][j]*alpha + v[i][j]*beta)
    img_out = (np.absolute(depth/np.linalg.norm(depth)*255.0)).astype(int)
    print(np.count_nonzero(img_out))
    print(img_out)
    img_out = scipy.ndimage.gaussian_filter(img_out,sigma = 3.0)
    scipy.misc.imshow(img_out)
    # plt.imshow(img_out,interpolation = 'nearest')
    # plt.show()


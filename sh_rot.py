import numpy as np

def project_sh_band2(n):
    c1 = np.sqrt(15./np.pi) / 2.
    c2 = np.sqrt(5./np.pi) / 4.
    return np.array([c1*n[0]*n[1],c1*n[1]*n[2],c1*n[0]*n[2],c2*(3.*n[2]*n[2]-1.),c1*(n[0]*n[0]-n[1]*n[1])/2.])

def sh_rot(R, coeffs):
    new_coeffs = np.empty_like(coeffs)
    new_coeffs[0] = coeffs[0]
    new_coeffs[1:4] = np.dot(R, coeffs[1:4])
    
    invA = np.zeros((5,5), dtype=np.float32)
    c1 = 2. * np.sqrt(np.pi / 15.)
    c2 = 2. * np.sqrt(np.pi / 5.)
    invA[0,1] = invA[1,0] = invA[1,4] = c1
    invA[0:2,2] = -c1
    invA[1,3] = c2
    invA[0,4] = invA[2,0] = invA[3,2] = invA[4,1] = 2. * c1
    
    r2 = 1. / np.sqrt(2.)
    bases = np.array([[1,0,r2,r2,0],[0,0,r2,0,r2],[0,1,0,r2,r2]], dtype=np.float32)
    bases = np.dot(R, bases)
    projected_bases = np.empty((5,5),dtype=np.float32)
    for i in range(5):
        projected_bases[:,i] = project_sh_band2(bases[:,i])
    
    new_coeffs[4:9] = np.dot(projected_bases, np.dot(invA, coeffs[4:9]))
    
    return new_coeffs

def calc_x_rot(theta):
    R = np.identity(3)
    R[1,1] = R[2,2] = np.cos(theta)
    R[2,1] = np.sin(theta)
    R[1,2] = -R[2,1]
    return R

def calc_y_rot(theta):
    R = np.identity(3)
    R[0,0] = R[2,2] = np.cos(theta)
    R[0,2] = np.sin(theta)
    R[2,0] = -R[0,2]
    return R

def calc_z_rot(theta):
    R = np.identity(3)
    R[0,0] = R[1,1] = np.cos(theta)
    R[1,0] = np.sin(theta)
    R[0,1] = -R[1,0]
    return R

if __name__ == '__main__':
    import sys
    import os
    import cv2
    
    if len(sys.argv) != 3:
        print('%s transport light' % sys.argv[0])
        exit(0)
    
    t_root, t_ext = os.path.splitext(sys.argv[1])
    
    T = np.load(sys.argv[1])
    if t_ext == '.npz':
        T = T['T']

    L = np.load(sys.argv[2])
    if L.shape[0] == 3:
        L = L.T
    
    n_div = 36
    for i in range(n_div):
        deg = (360. / n_div) * i
        R = calc_y_rot(deg / 180. * np.pi)
        coeffs = np.empty_like(L)
        coeffs[:,0] = sh_rot(R, L[:,0])
        coeffs[:,1] = sh_rot(R, L[:,1])
        coeffs[:,2] = sh_rot(R, L[:,2])
        I = np.matmul(T, coeffs)
        I = cv2.flip(I, 0)
        cv2.imwrite('sh_rot/rot%03d.jpg' % int(deg), 255 * I)

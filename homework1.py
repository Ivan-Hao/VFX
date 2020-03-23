import cv2 
import numpy as np
import os 
import math
from numpy import linalg as la

def alignment(base, **others):
    pass

def weighted(value):
    return 255-value if value > 127 else value

def response_curve(picture, t ,lamb):
    P = len(picture)
    sample_matrix = np.zeros(picture[0].shape[0:2])
    sample_matrix[::30,::30] = 1
    sample_index = np.where(sample_matrix == 1)
    sample_N = np.expand_dims(picture[0][:,:,0][sample_index], axis=0) 
    for i in range(1,P):
        sample_N = np.concatenate((sample_N,np.expand_dims( picture[i][:,:,0][sample_index],axis=0)), axis=0)
    N =  sample_N.shape[1] #sample points
    A = np.zeros((N*P+1+254,256+N))
    b = np.zeros((N*P+1+254,1))
    print(sample_N.shape,A.shape,b.shape)
    
    k = 0 
    for i in range(sample_N.shape[0]):
        for j in range(sample_N.shape[1]):
            wij = weighted(sample_N[i,j])
            A[k,sample_N[i,j]] = wij
            A[256+i] = -wij
            b[k,0] = wij * t[i]
            k+=1

    A[k,127] = 1
    k+=1
    for i in range(1,255):
        A[k,i-1] = lamb * weighted(i)
        A[k,i] = -2 * lamb * weighted(i)
        A[k,i+1] = lamb * weighted(i)
        k+=1
    x = la.lstsq(A,b,rcond=None)
    print(x[0])
    
    pass

def tone_mapping():
    pass



if __name__ == '__main__':
    lamb = 0.5
    base = '.\\img'
    img = []
    for root,dirs,files in os.walk(base):
        for i in files:
            img.append(cv2.imread(os.path.join(base,i)))
    explosure_time = list(map(lambda x: math.log(x), [2,1,0.5,0.25,0.125]))
    res_cur = response_curve(img,explosure_time,lamb)

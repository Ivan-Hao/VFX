import cv2 
import numpy as np
import os 
import math
from numpy import linalg as la
import matplotlib.pyplot as plt

def alignment(base, **others):
    pass

def weighted(value):
    return 255-value if value > 127 else value

def response_curve(picture, t ,lamb):
    P = len(picture)
    x_interval = picture[0].shape[0]//7
    y_interval = picture[0].shape[1]//7
    sample_matrix = np.zeros(picture[0].shape[0:2])
    sample_matrix[x_interval:-x_interval+1:x_interval,y_interval:-y_interval+1:y_interval] = 1
    sample_index = np.where(sample_matrix == 1)
    sample_B = np.expand_dims(picture[0][:,:,0][sample_index], axis=0) 
    sample_G = np.expand_dims(picture[0][:,:,1][sample_index], axis=0) 
    sample_R = np.expand_dims(picture[0][:,:,2][sample_index], axis=0) 
    for i in range(1,P):
        sample_B = np.concatenate((sample_B,np.expand_dims( picture[i][:,:,0][sample_index],axis=0)), axis=0)
        sample_G = np.concatenate((sample_G,np.expand_dims( picture[i][:,:,1][sample_index],axis=0)), axis=0)
        sample_R = np.concatenate((sample_R,np.expand_dims( picture[i][:,:,2][sample_index],axis=0)), axis=0)

    ret = []
    N =  sample_B.shape[1] #sample points
    sample = [sample_B,sample_G,sample_R]

    for z in range(3):
        A = np.zeros((N*P+1+254,256+N))
        b = np.zeros((N*P+1+254,1))
        
        k = 0 
        for i in range(sample[z].shape[0]):
            for j in range(sample[z].shape[1]):
                wij = weighted(sample[z][i,j])
                A[k,sample[z][i,j]] = wij
                A[k,256+j] = -wij
                b[k,0] = wij * t[i]
                k+=1

        A[k,127] = 1
        k+=1
        for i in range(1,255):
            A[k,i-1] = lamb * weighted(i)
            A[k,i] = -2 * lamb * weighted(i)
            A[k,i+1] = lamb * weighted(i)
            k+=1
        x = la.lstsq(A,b,rcond=None)[0]
        ret.append(x[0:256,0])
    return ret
    

def tone_mapping():
    pass



if __name__ == '__main__':
    lamb = 1
    base = '.\\test'
    img = []
    for root,dirs,files in os.walk(base):
        for i in files:
            img.append(cv2.imread(os.path.join(base,i)))
    explosure_time = list(map(lambda x: math.log(x),[4,2,1,0.5,0.25,0.125,0.0625,0.03125]))
    res_cur = response_curve(img,explosure_time,lamb)
    
    seq = [i for i in range(256)]
    plt.plot(res_cur[0],seq )
    plt.plot(res_cur[1],seq )
    plt.plot(res_cur[2],seq )
    plt.show()

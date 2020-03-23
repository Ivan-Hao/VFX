import cv2 
import numpy as np
import os 
import math

def alignment(base, **others):
    pass

def weighted(value):
    return 255-value if value > 127 else value

def response_curve(picture, t ,lamb):
    P = len(picture)
    N = 100 #sample point
    A = np.zeros((N*P+1+254,256+N))
    b = np.zeros((N*P+1+254,1))

    print(sample.shape)
    

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

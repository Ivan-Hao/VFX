from cv2 import cv2 
import numpy as np
import os
from numpy import linalg as la
import matplotlib.pyplot as plt

np.set_printoptions(threshold=20000000)
class HDR():

    def __init__(self, rgb_picture, gray_picture, base_picture, time, _lambda):
        self.rgb_picture = rgb_picture
        self.gray_picture =  gray_picture
        self.base_picture = base_picture 
        self.P = len(rgb_picture)
        self.shape = base_picture.shape[0:2]
        self._lambda = _lambda
        self.time = np.log(time,dtype=np.float64)
        self.res_curve = None
        self.irrandiance = None
        self.ln_irrandiance = None
        self.LDR = None


    def alignment(self):
        median = np.median(self.gray_picture[4])
        mask = np.logical_or(self.gray_picture[4] > median+10 , self.gray_picture[4] < median-10)
        print(mask)


        gray_base = self.gray_picture[4]
        base_pyramid = [gray_base[::2**i,::2**i] for i in range(8)]
        gray_pyramid = []
        for i in range(self.P):
            gray_pyramid.append([self.gray_picture[i][::2**j,::2**j] for j in range(8)])
        
        for i in range(self.P):
            shift_x=0
            shift_y=0
            for j in range(6,-1,-1):
                M = np.float64([[1, 0, shift_x], [0, 1, shift_y]])
                compare = cv2.warpAffine(gray_pyramid[i][j], M, gray_pyramid[i][j].shape[0:2][::-1])

                index = np.argmin([
                    (base_pyramid[j] != compare).sum(),
                    (compare[:,:-1] != base_pyramid[j][:,1:]).sum(), #上移
                    (compare[:,1:] != base_pyramid[j][:,:-1]).sum(), #下移
                    (compare[:-1,:] != base_pyramid[j][1:,:]).sum(), #左移
                    (compare[1:,:] != base_pyramid[j][:-1,:]).sum(), #右移
                    (compare[:-1,:-1] != base_pyramid[j][1:,1:]).sum(), #左上移
                    (compare[1:,1:] != base_pyramid[j][:-1,:-1]).sum(), #右下移
                    (compare[:-1,1:] != base_pyramid[j][1:,:-1]).sum(), #左下移
                    (compare[1:,:-1] != base_pyramid[j][:-1,1:]).sum(), #右上移
                ])
                print(shift_x,shift_y)
                
                if index == 1: # 上
                    shift_y -=1
                elif index == 2: # 下
                    shift_y +=1
                elif index == 3: # 左
                    shift_x -=1
                elif index == 4: # 右
                    shift_x +=1
                elif index == 5: #左上
                    shift_x -= 1
                    shift_y -= 1
                elif index == 6: #右下
                    shift_x += 1
                    shift_y += 1
                elif index == 7: #左下
                    shift_x -= 1
                    shift_y += 1
                elif index == 8: #右上
                    shift_x += 1
                    shift_y -= 1

                shift_y *=2
                shift_x *=2

            #cv2.imwrite('./result/result'+ str(i)+'.jpg', res)   



    def weighted(self, value):
        return 256-value if value > 127 else 1+value
    
    '''
    def sample_pattern(self,rgb):
        pattern = np.zeros(self.shape,dtype=bool)
        for i in range(256):
            r, c = np.where(self.base_picture[:,:,rgb] == i) 
            if len(r) !=0  :
                pattern[r[len(r)//2],c[len(c)//2]] = True
        return pattern
    '''

    def response_curve(self):
        
        x_interval = self.shape[0]//20
        y_interval = self.shape[1]//20
        sample_matrix = np.zeros(self.shape,dtype=bool)
        sample_matrix[x_interval:-x_interval+1:x_interval,y_interval:-y_interval+1:y_interval] = True
        
        sample_R = np.expand_dims(self.rgb_picture[0][:,:,0][sample_matrix], axis=1)
        sample_G = np.expand_dims(self.rgb_picture[0][:,:,1][sample_matrix], axis=1) 
        sample_B = np.expand_dims(self.rgb_picture[0][:,:,2][sample_matrix], axis=1)
 
        for i in range(1,self.P):
            sample_R = np.concatenate((sample_R,np.expand_dims( self.rgb_picture[i][:,:,0][sample_matrix],axis=1)), axis=1)
            sample_G = np.concatenate((sample_G,np.expand_dims( self.rgb_picture[i][:,:,1][sample_matrix],axis=1)), axis=1)
            sample_B = np.concatenate((sample_B,np.expand_dims( self.rgb_picture[i][:,:,2][sample_matrix],axis=1)), axis=1)

        ret = []
        sample = [sample_R,sample_G,sample_B]

        for z in range(3):
            N = sample[z].shape[0] #sample points
            A = np.zeros((N*self.P+1+254,256+N), dtype=np.float64)
            b = np.zeros((N*self.P+1+254,1), dtype=np.float64)
            k = 0 
            for i in range(sample[z].shape[0]):  
                for j in range(sample[z].shape[1]):  
                    wij = self.weighted(sample[z][i,j])
                    A[k,sample[z][i,j]] = wij
                    A[k,256+i] = -wij
                    b[k,0] = wij * self.time[j]
                    k+=1
            A[k,128] = 1
            k+=1
            for i in range(1,255):
                A[k,i-1] = self._lambda * self.weighted(i)
                A[k,i] = -2 * self._lambda * self.weighted(i)
                A[k,i+1] = self._lambda * self.weighted(i)
                k+=1
            x = la.lstsq(A,b,rcond=None)[0]

            ret.append(x[0:256,0])
        return ret
    
    def plot_res_curve(self):
        self.res_curve = self.response_curve()
        seq = [i for i in range(256)]
        plt.plot(self.res_curve[0],seq,'r')
        plt.plot(self.res_curve[1],seq,'g')
        plt.plot(self.res_curve[2],seq,'b')
        plt.show()

    def construct_irradiance(self):
        irmap = np.zeros(self.base_picture.shape,dtype=np.float64)
        for i in range(3):
            divisor , dividend = np.zeros(self.shape,dtype=np.float64), np.zeros(self.shape,dtype=np.float64)
            for l in range(self.P):
                W = np.where(self.rgb_picture[l][:,:,i]>127,256-self.rgb_picture[l][:,:,i],1+self.rgb_picture[l][:,:,i])
                divisor += W
                dividend += W * (self.res_curve[i][self.rgb_picture[l][:,:,i]] - self.time[l])

            irmap[:,:,i] = dividend / divisor
        self.irrandiance = np.exp(irmap)
        self.ln_irrandiance = irmap

    def plot_ln_irmap(self):
        for i in range(3):
            plt.imshow(self.ln_irrandiance[:,:,i],cmap='jet')
            plt.colorbar()
            plt.show()


    def global_tone_mapping(self,delta=1e-5,a=0.5):
        L_bar = np.exp(np.mean(np.log(delta+self.irrandiance)))
        L_m = (a/L_bar)*self.irrandiance
        L_d = L_m/(1+L_m)
        LDR = (L_d*255).astype(np.uint8)
        self.LDR = LDR
        plt.imshow(LDR)
        plt.show()

    def gamma_mapping(self):
        gamma = np.power(self.LDR/float(np.max(self.LDR)), 1.5)
        plt.imshow(gamma)
        plt.show()




if __name__ == '__main__':
    lamb = 10
    base = '.\\test3'
    rgb_img = []
    gray_img = []
    for root,dirs,files in os.walk(base):
        for i in files:
            rgb_img.append(cv2.imread(os.path.join(base,i))[:,:,::-1])
            gray_img.append(cv2.imread(os.path.join(base,i),cv2.IMREAD_GRAYSCALE))
    
    explosure_time = [1/0.03125,1/0.0625,1/0.125,1/0.25,1/0.5,1,1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256,1/512,1/1024]
    #explosure_time= [1/15,1/20,1/25,1/30,1/40,1/50,1/60,1/80,1/100]
    explosure_time = [13,10,4,3.2,1,0.8,1/3,1/4,1/60,1/80,1/320,1/400,1/1000]
    HDR_instance = HDR(rgb_img, gray_img, rgb_img[4], explosure_time, lamb)
    HDR_instance.plot_res_curve()
    HDR_instance.construct_irradiance()
    HDR_instance.plot_ln_irmap()
    #HDR_instance.alignment()
    HDR_instance.global_tone_mapping()
    HDR_instance.gamma_mapping()

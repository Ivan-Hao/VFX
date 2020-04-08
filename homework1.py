from cv2 import cv2 
import numpy as np
import os
from numpy import linalg as la
import matplotlib.pyplot as plt

np.set_printoptions(threshold=20000000)
class HDR():

    def __init__(self, rgb_picture, gray_picture, base_picture, gray_base, time, _lambda):
        self.rgb_picture = rgb_picture
        self.gray_picture =  gray_picture
        self.base_picture = base_picture 
        self.gray_base = gray_base
        self.P = len(rgb_picture)
        self.shape = base_picture.shape[0:2]
        self._lambda = _lambda
        self.time = np.log(time,dtype=np.float64)
        self.res_curve = None
        self.irrandiance = None
        self.ln_irrandiance = None
        self.LDR = None


    def alignment(self):
        base_median = np.median(self.gray_base)
        base_bit = np.where(self.gray_base>base_median,255,0).astype(np.uint8)
        base_pyramid = [base_bit[::2**i,::2**i] for i in range(5)]

        mask = np.logical_or(self.gray_base > base_median+10 , self.gray_base < base_median-10)
        mask_pyramid = [mask[::2**i,::2**i] for i in range(5)]
        print(mask.sum()/(self.shape[1]*self.shape[0]))

        crop = []
        for i in range(5):
            crop.append((base_pyramid[i].shape[0]//10,base_pyramid[i].shape[1]//10))
            
        compare_pyramid = []
        for i in range(self.P):
            compare_median = np.median(self.gray_picture[i])
            compare_bit = np.where(self.gray_picture[i]>compare_median,255,0).astype(np.uint8)
            compare_pyramid.append([compare_bit[::2**j,::2**j] for j in range(10)])
        
        for i in range(self.P):
            shift_x=0
            shift_y=0
            for j in range(4,-1,-1):
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                compare = cv2.warpAffine(compare_pyramid[i][j], M, compare_pyramid[i][j].shape[0:2][::-1])
                base_crop = base_pyramid[j][crop[j][0]:-crop[j][0],crop[j][1]:-crop[j][0]]
                mask_crop = mask_pyramid[j][crop[j][0]:-crop[j][0],crop[j][1]:-crop[j][0]]
                xor_min = np.inf
                for r in range(-1,2,1): #上下
                    for c in range(-1,2,1): #左右                 
                        T = np.float32([[1, 0, c], [0, 1, r]])
                        compare_shift = cv2.warpAffine(compare, T, compare.shape[0:2][::-1])[crop[j][0]:-crop[j][0],crop[j][1]:-crop[j][0]]
                        xor_loss = np.logical_xor(base_crop,compare_shift)[mask_crop].sum()
                        if xor_loss < xor_min:
                            xor_min = xor_loss
                            temp_x, temp_y = c , r
                shift_x += temp_x
                shift_y += temp_y
                shift_y *=2
                shift_x *=2

            F = np.float32([[1, 0, shift_x/2], [0, 1, shift_y/2]])
            write = cv2.warpAffine(self.rgb_picture[i], F, self.rgb_picture[i].shape[0:2][::-1])
            cv2.imwrite('.\\alignment_result\\'+ str(i)+'.jpg', write[:,:,::-1])  # rgb -> bgr 



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
        
        x_interval = self.shape[0]//7
        y_interval = self.shape[1]//7
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
        cv2.imwrite('.\\result\\radiance.hdr', self.irrandiance[:,:,::-1])

    def plot_ln_irmap(self):
        for i in range(3):
            plt.imshow(self.ln_irrandiance[:,:,i],cmap='jet')
            plt.colorbar()
            plt.show()

    def global_tone_mapping(self,delta=1e-6,a=0.5):
        L_bar = np.exp(np.mean(np.log(delta+self.irrandiance)))
        L_m = (a/L_bar)*self.irrandiance
        L_w = np.max(L_m)
        L_d = L_m*(1 + (L_m / (L_w *L_w)))/(1+L_m)
        LDR = (L_d*255).round().astype(np.uint8)
        self.LDR = LDR
        cv2.imwrite('.\\result\\LDR.jpg',self.LDR[:,:,::-1])
        plt.imshow(LDR)
        plt.show()
    
    def gaussian(self,a,s, L_m, threshold=0.01):
        blur = np.zeros(self.base_picture.shape+(s,))
        V_s = np.zeros(self.base_picture.shape+(s,))
        for i in range(0,s,2):
            now = cv2.GaussianBlur(L_m, (i*2+1, i*2+1), 0)
            next_ = cv2.GaussianBlur(L_m, ((i+1)*2+1, (i+1)*2+1), 0)
            Vs = np.abs((now-next_)/(((2**a)/(s**2))+now))
            blur[:,:,:,i] = now
            blur[:,:,:,i+1] = next_
            V_s[:,:,:,i] = Vs
        
        s_max = np.argmax(V_s>threshold,axis=3)
        s_max[np.where(s_max == 0)] = 1
        s_max -= 1
        
        return 



    def local_tone_mapping(self,delta=1e-6,a=0.5):
        L_bar = np.exp(np.mean(np.log(delta+self.irrandiance)))
        L_m = (a/L_bar)*self.irrandiance
        L_s = self.gaussian(1,8,L_m,0.01)
        L_d = L_m/(1+L_s)
        LDR = (L_d*255).round().astype(np.uint8)
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
    for files in os.listdir(base):
        rgb_img.append(cv2.imread(os.path.join(base,files))[:,:,::-1])
        gray_img.append(cv2.imread(os.path.join(base,files),cv2.IMREAD_GRAYSCALE))
    
    #explosure_time = [1/0.03125,1/0.0625,1/0.125,1/0.25,1/0.5,1,1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256,1/512,1/1024]
    #explosure_time= [1/15,1/20,1/25,1/30,1/40,1/50,1/60,1/80,1/100]
    explosure_time = [13,10,4,3.2,1,0.8,1/3,1/4,1/60,1/80,1/320,1/400,1/1000]
    HDR_instance = HDR(rgb_img, gray_img, rgb_img[0], gray_img[0], explosure_time, lamb)
    HDR_instance.plot_res_curve()
    HDR_instance.construct_irradiance()
    #HDR_instance.plot_ln_irmap()
    #HDR_instance.alignment()
    HDR_instance.global_tone_mapping()
    HDR_instance.local_tone_mapping()
    #HDR_instance.gamma_mapping()

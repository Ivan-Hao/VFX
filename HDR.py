from cv2 import cv2 
import numpy as np
import os
import argparse
from numpy import linalg as la
import matplotlib.pyplot as plt

class HDR():

    def __init__(self, rgb_picture, gray_picture, base_picture, gray_base, time, _lambda=10, gamma=1.5):
        self.rgb_picture = rgb_picture
        self.gray_picture =  gray_picture
        self.base_picture = base_picture 
        self.gray_base = gray_base
        self.P = len(rgb_picture)
        self.shape = base_picture.shape[0:2]
        self.time = np.log(time,dtype=np.float64)
        self._lambda = _lambda
        self.gamma = gamma
        self.res_curve = None
        self.irradiance = None
        self.ln_irradiance = None
        self.LDR = None

    def alignment(self):
        base_median = np.median(self.gray_base)
        base_bit = np.where(self.gray_base>base_median,255,0).astype(np.uint8)
        base_pyramid = [base_bit[::2**i,::2**i] for i in range(6)]

        mask = np.logical_or(self.gray_base > base_median+10 , self.gray_base < base_median-10)
        mask_pyramid = [mask[::2**i,::2**i] for i in range(6)]

        
        length = 0
        for i in range(len(base_pyramid)):
            length+=base_pyramid[i].shape[1]
        x = np.ones((base_pyramid[0].shape[0],length))*255
        y = np.ones((base_pyramid[0].shape[0],length))*255
        temp=0
        for i in range(len(base_pyramid)):
            x[-base_pyramid[i].shape[0]:,temp:temp+base_pyramid[i].shape[1]] = base_pyramid[i]
            y[-mask_pyramid[i].shape[0]:,temp:temp+mask_pyramid[i].shape[1]] = mask_pyramid[i]*255
            temp +=base_pyramid[i].shape[1]
        plt.imshow(x,cmap='gray')
        plt.show()
        plt.imshow(y,cmap='gray')
        plt.show()
        

        crop = []
        for i in range(6):
            crop.append((base_pyramid[i].shape[1]//10,base_pyramid[i].shape[0]//10))
            
        compare_pyramid = []
        for i in range(self.P):
            compare_median = np.median(self.gray_picture[i])
            compare_bit = np.where(self.gray_picture[i]>compare_median,255,0).astype(np.uint8)
            compare_pyramid.append([compare_bit[::2**j,::2**j] for j in range(10)])
        
        for i in range(self.P):
            shift_x=0
            shift_y=0
            for j in range(5,-1,-1):
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                compare = cv2.warpAffine(compare_pyramid[i][j], M, compare_pyramid[i][j].shape[0:2][::-1])
                base_crop = base_pyramid[j][crop[j][0]:-crop[j][0],crop[j][1]:-crop[j][1]]
                mask_crop = mask_pyramid[j][crop[j][0]:-crop[j][0],crop[j][1]:-crop[j][1]]
                xor_min = np.inf
                for r in range(-1,2,1): #上下
                    for c in range(-1,2,1): #左右                 
                        T = np.float32([[1, 0, c], [0, 1, r]])
                        compare_shift = cv2.warpAffine(compare, T, compare.shape[0:2][::-1])[crop[j][0]:-crop[j][0],crop[j][1]:-crop[j][1]]
                        xor_loss = np.logical_xor(base_crop,compare_shift)[mask_crop].sum()
                        if xor_loss < xor_min:
                            xor_min = xor_loss
                            temp_x, temp_y = c , r
                shift_x += temp_x
                shift_y += temp_y
                shift_y *=2
                shift_x *=2
            print(shift_x/2,shift_y/2)
            F = np.float32([[1, 0, shift_x/2], [0, 1, shift_y/2]])
            write = cv2.warpAffine(self.rgb_picture[i], F, self.rgb_picture[i].shape[0:2][::-1])
            cv2.imwrite('.\\alignment_result\\'+ str(i)+'.jpg', write[:,:,::-1])  # rgb -> bgr 

    def weighted(self, value):
        return 256-value if value > 127 else 1+value
    
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
        self.res_curve = ret
    
    def plot_res_curve(self):
        seq = [i for i in range(256)]
        rgb = ['red','green','blue']
        for i in range(3):
            plt.plot(self.res_curve[i],seq,rgb[i])
            plt.title(rgb[i])
            plt.xlabel("ln E")
            plt.ylabel("pixel value")
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
        self.irradiance = np.exp(irmap)
        self.ln_irradiance = irmap
        cv2.imwrite('.\\result\\radiance.hdr', self.irradiance[:,:,::-1])

    def plot_ln_irmap(self):
        rgb = ['red','green','blue']
        for i in range(3):
            plt.imshow(self.ln_irradiance[:,:,i],cmap='jet')
            plt.title(rgb[i])
            plt.colorbar()
            plt.show()

    def global_tone_mapping(self,delta=1e-6,a=0.5):
        L_bar = np.exp(np.mean(np.log(delta+self.irradiance)))
        L_m = (a/L_bar)*self.irradiance
        L_w = np.max(L_m)
        L_d = L_m*(1 + (L_m / (L_w *L_w)))/(1+L_m)
        LDR = (L_d*255).round().astype(np.uint8)
        self.LDR = LDR
        cv2.imwrite('.\\result\\LDR.jpg',self.LDR[:,:,::-1])
        plt.imshow(LDR)
        plt.title("global tone mapping")
        plt.show()
    
    def gamma_mapping(self):
        gamma = np.zeros_like(self.LDR)
        for i in range(3):
            gamma[:,:,i] = (np.power(self.LDR[:,:,i]/float(np.max(self.LDR[:,:,i])),self.gamma)*255).round().astype(np.uint8)
        cv2.imwrite('.\\result\\gamma.jpg',gamma[:,:,::-1])
        plt.imshow(gamma)
        plt.title("gamma correction")
        plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", help="alignment the image", default=False,type=bool)
    parser.add_argument("--lambda_", help="function smooth parameter", default=10,type=int)
    parser.add_argument("--path", help="image dir path(image only)", default='./alignment_result',type=str)
    parser.add_argument("--index", help="The based picture's index(>=0)", default=0,type=int)
    parser.add_argument("--time", help="The explosure time file path(.txt or .npy)", default='Test_explosure.txt',type=str)
    parser.add_argument("--gamma", help="gamma parameter", default=1.5, type=float)
    
    args = parser.parse_args()
    
    rgb_img = []
    gray_img = []
    for files in os.listdir(args.path):
        rgb_img.append(cv2.imread(os.path.join(args.path,files))[:,:,::-1])
        gray_img.append(cv2.imread(os.path.join(args.path,files),cv2.IMREAD_GRAYSCALE))

    explosure_time =np.loadtxt(args.time) 
    HDR_instance = HDR(rgb_img, gray_img, rgb_img[args.index], gray_img[args.index], explosure_time, args.lambda_, args.gamma)
    if args.alignment:
        HDR_instance.alignment()
    else:
        HDR_instance.response_curve()
        HDR_instance.construct_irradiance()
        HDR_instance.plot_res_curve()
        HDR_instance.plot_ln_irmap()
        HDR_instance.global_tone_mapping()
        HDR_instance.gamma_mapping()

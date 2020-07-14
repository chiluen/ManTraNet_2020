import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.util import random_noise
from PIL import Image, ImageOps
import os


class preprocess_rand():
    """
    Usage:
    p = preprocess_rand('temp dir for saving compression pic temporaily')
    p = p.n_preprocess(img) # use three modification methods
    """
    
    def __init__(self, temp_output_dir):
        self.dic =  {0:self.img_blur, 1:self.img_erosion, 2:self.img_dilate, 3:self.img_noise, 
                     4:self.img_quantization, 5:self.img_autocontrast, 6:self.img_histeq, 7:self.img_compression}
        self.temp_output_dir = temp_output_dir
        
    def n_preprocess(self, img, n = 3):
        number = np.random.choice(np.arange(7), n,replace=False)
        compression_flag = False
        for i in range(n):
            img = self.dic[number[i]](img)

            
        return img
        
    # All the function
    def img_blur(self, img, kernel=(5,5)):
        blur = cv.blur(img, kernel)
        return blur
    
    def img_erosion(self, img, kernel=(5,5)):
        kernel = np.ones((5,5), np.uint8)
        erosion = cv.erode(img, kernel, iterations = 2)
        return erosion
    
    def img_dilate(self, img, kernel=(5,5)):
        kernel = np.ones((5,5), np.uint8)
        dilate = cv.dilate(img, kernel, iterations = 2)
        return dilate
    
    def img_noise(self, img, mode = 'gaussian'):
        noise_img = random_noise(img, mode = mode, var = 0.05**2)
        noise_img = (255*noise_img).astype(np.uint8)
        return noise_img    

    def img_quantization(self, img):
        
        Z = img.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        return res2

    def img_autocontrast(self, img):
        img = Image.fromarray(img)
        img = ImageOps.autocontrast(img, cutoff=10)
        return np.asarray(img)

    def img_histeq(self, img):
        img = img.copy()  #解決flag問題
        for i in range(3): #對RGB做histeq
            dst = cv.equalizeHist(img[:,:,i])
            img[:,:,i] = dst
        return img

    #只能在最後用來儲存照片時呼叫
    def img_compression(self, img, temp_output_dir = None):
        temp_output_dir = self.temp_output_dir
        cv.imwrite( os.path.join(temp_output_dir, 'temp.png'), img, [cv.IMWRITE_PNG_COMPRESSION, 5])
        img = cv.imread(os.path.join(temp_output_dir, 'temp.png'))
        return img
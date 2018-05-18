import os,sys,pdb
import numpy as np
import cv2

class DATA_AUGMENT(object):
    def __init__(self):
        return
    def postion(self, img, w, h):
        x0 = np.int64(random.random() * (img.shape[1] - w) / 2)
        y0 = np.int64(random.random() * (img.shape[0] - h) / 2)
        return img[y0:y0+h, x0:x0+w,:]
        
    def color(self, img):
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
        hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7)
        hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8)
        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    def gaussian(self,img):
        sig = random.random() * 1.5
        ks = np.int64(sig * 3/2)
        ks = np.minimum( ks, img.shape[0] / 10 )
        ks = np.maximum( ks, img.shape[1] / 5 )
        ks = np.maximum( ks, 1)
        ks = ks * 2 + 1
        return cv2.GaussianBlur(img, (ks,ks), sig)
    def add_noise_gray(self,gray):
        diff = 255-gray.max();
        noise = np.random.normal(0,1+random.random()*6, gray.shape);
        noise = (noise - noise.min())/(noise.max()-noise.min())
        noise= diff*noise;
        noise= noise.astype(np.uint8)
        dst = gray + noise
        return dst   
    def add_noise(self, img):
        img[:,:,0] = self.add_noise_gray(img[:,:,0])
        img[:,:,1] = self.add_noise_gray(img[:,:,1])
        img[:,:,2] = self.add_noise_gray(img[:,:,2])
        return img
        
      
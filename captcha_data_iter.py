import os,sys,pdb,cPickle
import numpy as np
import mxnet as mx
import cv2
from data_augment import DATA_AUGMENT
import random

class CAPTCHABatch(mx.io.DataBatch):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]
        

class CAPTCHAIter(mx.io.DataIter):
    def load_samples(self, root,sampleFile):
        samples = []
        with open(sampleFile, "rb") as f:
            for line in f:
                line = line.strip().split('|')
                label =  [ self.char2idx[x] for x in list( line[0] )] 
                path = os.path.join(root,line[1])
                samples.append( (path,label) )
        return samples
    def calc_mean(self):
        redM, greenM, blueM = [], [], []
        redS, greenS, blueS = [], [], []
        for (path,label) in self.samples:
            img = np.float64( cv2.imread( path, 1) )
            redM.append( img[:,:,0].mean() )
            greenM.append( img[:,:,1].mean() )
            blueM.append( img[:,:,2].mean() )
            redS.append( img[:,:,0].std() )
            greenS.append( img[:,:,1].std() )
            blueS.append( img[:,:,2].std() )
        #pdb.set_trace()
        redM = np.asarray( redM ).mean()
        blueM = np.asarray(blueM).mean()
        greenM = np.asarray(greenM).mean()
        
        redS = np.asarray( redS ).mean()
        blueS = np.asarray(blueS).mean()
        greenS = np.asarray(greenS).mean()
        return (blueM,greenM,redM),(blueS, greenS, redS)
        
    def get_sample(self, idx, path = None):
        if path is None:
            path, label = self.samples[idx]
        else:
            label = '0000'
        img = cv2.imread(path,1) 
        #cv2.imshow('src',img)
        #pdb.set_trace()
        if self.dataAug:
            da = DATA_AUGMENT()
            img = da.gaussian(img)
            img = da.add_noise(img)
            img = da.postion(img, self.width, self.height)
        else:
            if self.width < img.shape[1] or self.height < img.shape[0]:
                cx,cy = img.shape[1] / 2,img.shape[0] / 2
                x0 = cx - self.width/2
                x1 = x0 + self.width
                y0 = cy - self.height/2
                y1 = y0 + self.height
                img = img[y0:y1,x0:x1,:]
        
        #cv2.imshow('new',img)
        #cv2.waitKey(200)
        #pdb.set_trace()
        blueM,greenM,redM = self.bgrMean
        blueS,greenS,redS = self.bgrStd
        img = img * 1.0
        #pdb.set_trace()
        img[:,:,0] = (img[:,:,0] - blueM) / blueS
        img[:,:,1] = (img[:,:,1] - greenM) / greenS
        img[:,:,2] = (img[:,:,2] - redM) / redS
        img = np.transpose(img, (2,0,1)) ###### transpose
        return img,label
                
    def __init__(self, root, sampleFile,batch_size, labels, width, height, shuffle = False,dataAug = False, initMeanStd=False):
        super(CAPTCHAIter, self).__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataAug = dataAug
        self.height = height
        self.width = width
        self.num_label = len(labels)
        self.char2idx = {}
        self.idx2char = {}
        labelnames = labels
        for idx,name in enumerate(labelnames):
            self.char2idx[name] = idx
            self.idx2char[idx] = name
        self.samples = self.load_samples(root,sampleFile)
        if initMeanStd == True:
            self.bgrMean, self.bgrStd = self.calc_mean()
            with open('mean_std.pkl','wb') as f:
                cPickle.dump((self.bgrMean,self.bgrStd), f)
        else:
            with open('mean_std.pkl','rb') as f:
                self.bgrMean,self.bgrStd = cPickle.load(f)
        if shuffle is True:
            random.shuffle( self.samples )
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, self.num_label))]
    def __iter__(self):
        for k in range( len(self.samples) / self.batch_size ):
            data = []
            label = []
            for i in range(self.batch_size):
                X,Y = self.get_sample(i + k * self.batch_size)
                data.append(X)
                label.append(Y)

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']
            data_batch = CAPTCHABatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        if self.shuffle:
            random.shuffle( self.samples )
        pass
  
import os,sys,pdb
from mxnet.gluon import nn
import mxnet as mx
import logging
class CAPTCHANet(nn.Block):
    def __init__(self,outputNum,verbose = False, **kwargs):
        super(CAPTCHANet,self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            layers = []
            layers.append( nn.Conv2D(32, kernel_size = 3, strides = 1, padding = 0, activation='relu') )
            layers.append( nn.Conv2D(32, kernel_size = 3, strides = 1, padding = 0, activation='relu') )
            layers.append( nn.Conv2D(32, kernel_size = 3, strides = 1, padding = 0, activation='relu') )
            layers.append( nn.MaxPool2D(pool_size=3, strides = 2) )
            layers.append( nn.Conv2D(64, kernel_size = 3, strides = 1, padding = 0, activation='relu') )
            layers.append( nn.Conv2D(64, kernel_size = 3, strides = 1, padding = 0, activation='relu') )
            layers.append( nn.Conv2D(64, kernel_size = 3, strides = 1, padding = 0, activation='relu') )
            layers.append( nn.MaxPool2D(pool_size=3, strides = 2) )
            layers.append( nn.Dense(2048,activation='relu') )
            layers.append( nn.Dense(1024,activation='relu') )
            self.netbody = nn.Sequential()
            for layer in layers:
                self.netbody.add(layer)
            self.netout = nn.Sequential()
            self.netout.add(nn.Dense(outputNum,activation='relu'))
            self.netout.add(nn.Dense(outputNum,activation='relu'))
            self.netout.add(nn.Dense(outputNum,activation='relu'))
            self.netout.add(nn.Dense(outputNum,activation='relu'))
        return
    def forward(self,X):
        bodyout = X
        for i,layer in enumerate(self.netbody):
            bodyout = layer(bodyout)
            if self.verbose:
                print 'bodylayer#',i+1,' shape:',bodyout.shape
        out = []
        for i,layer in enumerate(self.netout):
            tmp =  layer(bodyout) 
            if self.verbose:
                print 'outlayer#',i+1,' shape:',tmp.shape
            out.append(tmp)
        return out

def get_symbol(outputNum,ctx): 
    net = CAPTCHANet(outputNum)
    net_str = '%s'%net
    logging.info(net_str)
    net.initialize(ctx = ctx)
    return net
     
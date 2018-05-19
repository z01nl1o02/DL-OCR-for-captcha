import os,sys,pdb,cPickle
import numpy as np
import cv2
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
import random
import math
from mxnet.gluon import nn
import utils
import logging
from importlib import import_module
from captcha_data_iter import CAPTCHAIter

pretrained=""

dataroot = 'data/qq'
labels = list('abcdefghijklmnopqrstuvwxyz')
outputNum = len(labels)
trainBatchSize = 50
testBatchSize = 50
width = 168
height = 64
ctx = mx.gpu()

   
logging.basicConfig(format='%(asctime)s %(message)s', filemode='w',datefmt='%m/%d/%Y %I:%M:%S %p',filename="train.log", level=logging.INFO)


mod = import_module('symbols.captcha_resnet')
net = mod.get_symbol(outputNum, ctx)



if pretrained is not None and pretrained != "":
    net.load_params(pretrained,ctx=utils.try_gpu())
    logging.info("load model:%s"%pretrained)


trainIter = CAPTCHAIter(dataroot,os.path.join(dataroot,'train/samples.txt'), trainBatchSize,labels, width, height, shuffle=True, dataAug = True, initMeanStd = True)
testIter = CAPTCHAIter(dataroot,os.path.join(dataroot,'test/samples.txt'), testBatchSize, labels, width, height)
for batch in testIter:
    data,label = batch.data, batch.label
    print data[0].shape, label[0].shape
    break 

    
loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label = True, axis=1)    
trainer = gluon.Trainer(net.collect_params(),"adam",{'learning_rate':0.001,'wd':0.00005})
    
    
#lr_steps = [k * 100 for k in [500, 2000, 3000, 4000]]
lr_steps = [10000000]
utils.train(trainIter, testIter, net, loss, trainer, ctx, lr_steps[-1] + 1000, lr_steps, print_batches = 200, cpdir = "model")
    
    
    
        
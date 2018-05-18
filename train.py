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

pretrained="model/Aiter-072624.params"

dataroot = 'data/gen'
labels = list('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
outputNum = len(labels)
trainBatchSize = 50
testBatchSize = 50
width = 120
height = 32
ctx = mx.gpu()

   
logging.basicConfig(format='%(asctime)s %(message)s', filemode='w',datefmt='%m/%d/%Y %I:%M:%S %p',filename="train.log", level=logging.INFO)


mod = import_module('symbols.captcha_net')
net = mod.get_symbol(outputNum, ctx)



if pretrained is not None and pretrained != "":
    net.load_params(pretrained,ctx=utils.try_gpu())
    logging.info("load model:%s"%pretrained)

testIter = CAPTCHAIter(dataroot,os.path.join(dataroot,'test/samples.txt'), testBatchSize, labels, width, height)
trainIter = CAPTCHAIter(dataroot,os.path.join(dataroot,'train/samples.txt'), trainBatchSize,labels, width, height, shuffle=True, dataAug = True, initMeanStd = True)

#for batch in testIter:
#    data,label = batch.data, batch.label
#    print data[0].shape, label[0].shape

loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label = True, axis=1)    
trainer = gluon.Trainer(net.collect_params(),"sgd",{'learning_rate':0.01,'wd':0.00005})
    
    
lr_steps = [k * 100 for k in [1000, 2000, 3000, 4000]]
utils.train(trainIter, testIter, net, loss, trainer, ctx, lr_steps[-1] + 1000, lr_steps, print_batches = 200, cpdir = "model")
    
    
    
        
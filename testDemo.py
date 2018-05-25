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
from captcha_data_iter import CAPTCHAIter
from data_augment import DATA_AUGMENT
from importlib import import_module
import glob

pretrained="model/weights.params"

dataroot = 'data/qq/test' #has test subdir
labels = list('abcdefghijklmnopqrstuvwxyz')
outputNum = len(labels)
testBatchSize = 1
width = 168
height = 64
ctx = mx.gpu()




mod = import_module('symbols.captcha_resnet')
net = mod.get_symbol(outputNum, ctx)
net.load_params(pretrained,ctx=ctx)

testIter = CAPTCHAIter(dataroot,os.path.join(dataroot,'samples.txt'), testBatchSize, labels, width, height)

lossFunc = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label = True, axis=1)    

for jpg in glob.glob( os.path.join(dataroot,'*.jpg') ):
    img,label = testIter.get_sample(0, jpg)
    X = nd.array( np.expand_dims(img,0) ).as_in_context(ctx)
    Y = net(X)
    chars = [ labels[np.uint64(y.argmax(axis=1).asnumpy()[0])] for y in Y]
    print chars
    cv2.imshow('img',cv2.imread(jpg,1))
    cv2.waitKey(-1)
    
        
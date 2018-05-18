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
import glob 

pretrained="model/Aiter-072624.params"

dataroot = 'data/gen/test' #has test subdir
labels = list('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
outputNum = len(labels)
testBatchSize = 50
width = 120
height = 32
ctx = mx.cpu()


mod = import_module('symbols.captcha_net')
net = mod.get_symbol(outputNum,ctx)
net.load_params(pretrained,ctx=ctx)

testIter = CAPTCHAIter(dataroot,os.path.join(dataroot,'samples.txt'), testBatchSize, labels, width, height)


lossFunc = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label = True, axis=1)    



acc, loss = utils.evaluate_accuracy(testIter, net, lossFunc,ctx=ctx)
print 'acc %f loss %f'%(acc,loss)
        
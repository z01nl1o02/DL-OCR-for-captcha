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
from captcha_data_iter import train_dataset, test_dataset
import datetime


pretrained=""#"models/epoch_{:0>8d}.params".format(19)

batch_size = 32
outputNum = 26
ctx = mx.gpu()


mod = import_module('symbols.captcha_resnet')
net = mod.get_symbol(outputNum, ctx)



if pretrained is not None and pretrained != "":
    net.load_params(pretrained,ctx=utils.try_gpu())
    logging.info("load model:%s"%pretrained)

train_iter = gluon.data.DataLoader(train_dataset,batch_size, shuffle=True,last_batch="rollover")
test_iter = gluon.data.DataLoader(test_dataset,batch_size, shuffle=False,last_batch="rollover")


base_lr = 0.001
max_epoch = 100
max_update = max_epoch * len(train_dataset) / batch_size
loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label = True, axis=1)    
trainer = gluon.Trainer(net.collect_params(),"adam",{'learning_rate':base_lr,'wd':0.0005})
lr_sch = mx.lr_scheduler.PolyScheduler(max_update,base_lr=base_lr,pwr=1)
    
utils.train(train_iter, test_iter, net, loss, trainer, ctx, max_epoch, len(train_dataset)//batch_size//10, lr_sch)
    

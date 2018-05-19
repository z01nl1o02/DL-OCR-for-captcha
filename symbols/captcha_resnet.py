#!/usr/bin/env python3
# -*- encoding:utf-8 -*-

"""
source: https://github.com/ZhichengHuang/MxNet_BoT
"""

from mxnet.gluon import nn


class BasicBlock(nn.HybridBlock):
    def __init__(self,channels,same_shape=True,first_layer=False,**kwargs):
        super(BasicBlock,self).__init__(**kwargs)
        self.expansion = 1
        self.same_shape= same_shape
        strides = 1 if same_shape else 2

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels,kernel_size=3,padding=1,
                                   strides=strides,use_bias=False))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Conv2D(channels,kernel_size=3,padding=1,use_bias=False))
        self.body.add(nn.BatchNorm())

        if not same_shape:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels,kernel_size=1,strides=strides,
                                       use_bias=False))


    def hybrid_forward(self, F, x):
        out = self.body(x)

        if  not self.same_shape:
            x = self.downsample(x)
        return F.relu(out+x)


class Bottleneck(nn.HybridBlock):
    def __init__(self,channels,same_shape=True,first_layer=False,**kwargs):
        super(Bottleneck,self).__init__(**kwargs)
        self.expansion = 4
        self.same_shape=same_shape
        strides = 1 if same_shape or first_layer else 2

        self.body = nn.HybridSequential(prefix="")
        self.body.add(nn.Conv2D(channels,kernel_size=1,strides=strides))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))

        self.body.add(nn.Conv2D(channels,kernel_size=3,padding=1,
                                   use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))

        self.body.add(nn.Conv2D(channels*self.expansion,kernel_size=1))
        self.body.add(nn.BatchNorm())

        if (not same_shape) or first_layer:

            self.downsame=nn.HybridSequential(prefix='')
            self.downsame.add(nn.Conv2D(channels * self.expansion, kernel_size=1,
                                   strides=strides, use_bias=False))
            self.downsame.add(nn.BatchNorm())
        else:
            self.downsame=None


    def hybrid_forward(self, F, x):
        out = self.body(x)

        if (not self.same_shape)and self.downsame:
            x = self.downsame(x)
        return F.relu(x+out)







class ResNet(nn.HybridBlock):
    def __init__(self,block,layers,last_pool=True,num_classes=10, verbose=False,**kwargs):
        super(ResNet,self).__init__(**kwargs)
        self.verbose=verbose
        self.last_pool=last_pool
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3,use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(pool_size=3,strides=2,padding=1))

            self.features.add(self._make_layer(block,64,layers[0],same_shape=False,
                                           first_layer=True))

            self.features.add(self._make_layer(block,128,layers[1],same_shape=False))

            self.features.add(self._make_layer(block,256,layers[2],same_shape=False))

            self.features.add(self._make_layer(block,512,layers[3],same_shape=False))

            if self.last_pool:
                self.classifier = nn.HybridSequential(prefix='')
                self.classifier.add(nn.GlobalAvgPool2D())
                self.classifier.add(nn.Dense(1024))
            else:
                self.classifier=None

            self.netout = nn.Sequential()
            self.netout.add(nn.Dense(outputNum))
            self.netout.add(nn.Dense(outputNum))
            self.netout.add(nn.Dense(outputNum))
            self.netout.add(nn.Dense(outputNum))

    def hybrid_forward(self, F, x):
        bodyout = self.features(x)
        if self.classifier:
            bodyout=self.classifier(bodyout)
        out = []
        for i,layer in enumerate(self.netout):
            tmp =  layer(bodyout) 
            out.append(tmp)
        return out



    def _make_layer(self,block,channels,blocks,same_shape=True,first_layer=False):
        bottleLay = nn.HybridSequential(prefix='')
        with bottleLay.name_scope():
            bottleLay.add(block(channels, same_shape=same_shape, first_layer=first_layer,prefix=''))

            for i in range(1, blocks):
                bottleLay.add(block(channels,prefix=''))

        return bottleLay


def get_symbol(num_classes,ctx,pretrained=False,verbose=False,**kwargs):
    net = ResNet(Bottleneck,[3,4,6,3],last_pool=True,
                   num_classes=num_classes,verbose=verbose,**kwargs)
    net.initialize(ctx = ctx)
    return net

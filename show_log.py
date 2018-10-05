import os,sys
import numpy as np
import re
from matplotlib import pyplot as plt


class TRACE:
    def __init__(self,name,color,scale=1):
        self.data_list = []
        self.name = name
        self.color = color
        self.scale = scale
        return
    def update(self,update,value):
        self.data_list.append( (int(update), float(value) * self.scale ) )
        return
    def plot(self):
        X = map(lambda x:x[0], self.data_list)
        Y = map(lambda x:x[1], self.data_list)
        plt.plot(X,Y,label=self.name,color=self.color)


def show_log(filepath):
    train_acc = TRACE("train_acc",(1,0,0))
    train_loss = TRACE('train_loss',(1,0.5,0))
    test_acc = TRACE("test_acc",(0,0,1))
    test_loss = TRACE("test_loss",(0,0.5,1))
    lr_trace = TRACE("lr-1000x",(0,1,0),scale=1024)
    last_update = 0
    with open(filepath,'rb') as f:
        for line in f:
            if re.findall("train epoch",line):
                update = re.findall(r"update:(\S+)",line)[0]
                lr = re.findall(r"lr:(\S+)",line)[0]
                loss = re.findall(r'loss:(\S+)',line)[0]
                acc = re.findall(r'acc:(\S+)',line)[0]
                last_update = update
                train_acc.update(update,acc)
                train_loss.update(update,loss)
                lr_trace.update(update,lr)
            if re.findall("test epoch",line):
                update = last_update
                loss = re.findall(r'loss:(\S+)',line)[0]
                acc = re.findall(r'acc:(\S+)',line)[0]
                test_acc.update(update,acc)
                test_loss.update(update,loss)
    fig = plt.figure()
    train_acc.plot()
    train_loss.plot()
    test_acc.plot()
    test_loss.plot()
    lr_trace.plot()
    plt.ylim(0,1)
    plt.legend()
    fig.savefig("train-test-loss.png")


if __name__ == "__main__":
    show_log(sys.argv[1])






    


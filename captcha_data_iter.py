import os,cv2
import numpy as np
from mxnet import gluon
from data_augment import DATA_AUGMENT
import random
import logging
import datetime


ratio_for_train = 0.9
root = "c:/dataset/captcha/sample/"
input_size = (32,84)
resample_rate = 1



nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
if not os.path.exists('log'):
    os.makedirs('log')
handleFile = logging.FileHandler("log/log.%s.txt"%nowTime,mode="wb")
handleFile.setFormatter(formatter)

handleConsole = logging.StreamHandler()
handleConsole.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
logger.handles = []
logger.addHandler(handleFile)
logger.addHandler(handleConsole)



class CaptchaDataset(gluon.data.Dataset):
    def __init__(self,file_list, size ,fortrain,*kwargs):
        self.fortrain = fortrain
        self.size = size
        self.label2idx,self.idx2label = {},{}
        for idx,label in enumerate( list('abcdefghijklmnopqrstuvwxyz') ):
            self.label2idx[label] = idx
            self.idx2label[idx] = label
        self.data_list = []
        if isinstance(file_list,str):
            with open(file_list,'rb') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    name = os.path.splitext(os.path.split(line)[-1])[0]
                    labels = map(lambda x:self.label2idx[x], list(name))
                    self.data_list.append((line,labels))
        else:
            for path in file_list:
                name = os.path.splitext(os.path.split(path)[-1])[0]
                labels = map(lambda x:self.label2idx[x], list(name))
                self.data_list.append((path,labels))
        logging.info("fortrain {} {}".format(self.fortrain,len(self.data_list)))
        return
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, item):
        path,labels = self.data_list[item]
        img = cv2.imread(path,1)
        H,W,C = img.shape
        if H != self.size[0] or W != self.size[1]:
            img = cv2.resize(img,(self.size[1],self.size[0]))
        img = np.transpose(img,(2,0,1))
        img = np.float32(img / 255.0)
        return img,labels

path_list = []
if not os.path.exists('train.txt') or not os.path.exists('test.txt'):
    for path in os.listdir(root):
        if os.path.splitext(path)[-1] != '.jpg':
            continue
        path_list.append( os.path.join(root,path))
    random.shuffle(path_list)
    path_list = path_list[0:int(len(path_list)*resample_rate)]
    num_for_train = int(len(path_list) * ratio_for_train)
    train_list = path_list[0:num_for_train]
    test_list = path_list[num_for_train:]
    with open('train.txt','wb') as f:
        f.write('\n'.join(train_list))
    with open('test.txt','wb') as f:
        f.write('\n'.join(test_list))
    train_dataset = CaptchaDataset(train_list,input_size,True)
    test_dataset = CaptchaDataset(test_list,input_size,False)
else:
    train_dataset = CaptchaDataset("train.txt",input_size,True)
    test_dataset = CaptchaDataset('test.txt',input_size,False)




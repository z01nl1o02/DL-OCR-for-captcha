import os,sys,pdb
import numpy as np
from PIL import Image
import shutil
import random

def save_to(imgs, outdir):
    try:
        os.makedirs(outdir)
    except Exception,e:
        pass
    for img in imgs:
        shutil.move( img, outdir)
    return 

def run(inroot, outroot, ratio = 0.3):
    train = []
    test = []
    imgs = []
    for root, pdirs, names in os.walk(inroot):
        for name in names:
            sname,ext = os.path.splitext(name)
            if ext != '.jpg':
                continue
            imgs.append( os.path.join(root, name) )
    random.shuffle( imgs )
    N = np.int64(len(imgs) * ratio)
    train = imgs[N:]
    test = imgs[0:N]
    save_to( train, os.path.join(outroot,'train/'))
    save_to( test, os.path.join(outroot, 'test/'))
    
if __name__=="__main__":
    run(sys.argv[1], sys.argv[1])
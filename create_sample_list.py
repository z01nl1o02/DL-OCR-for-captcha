import os,sys,pdb
import numpy as np

root = 'data/qq'

def run(inroot,relpath):
    lines = []
    jpgs = os.listdir(os.path.join(inroot,relpath))
    #pdb.set_trace()
    for jpg in jpgs:
        sname,ext = os.path.splitext(jpg)
        if ext != '.jpg':
            continue
        path = os.path.join(relpath,jpg)
        lines.append( "|".join([sname,path]))
    with open(os.path.join(os.path.join(inroot,relpath),'samples.txt'),"wb") as f:
        f.write('\r\n'.join(lines))
    return 
    
run(root,'train' )
run(root,'test' )
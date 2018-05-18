
'''
generate simplest captcha 
'''

import numpy as np
import cv2
import random,os,sys

if __name__ == '__main__':
    print __doc__
    outdir = '../data/gen'
    try: 
        os.makedirs(outdir)
    except Exception,e:
        pass
    H,W = 32,128
    left, bottom = 10, H - 5
    font = cv2.FONT_HERSHEY_COMPLEX
    idx_to_char = list('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
    charNum = len(idx_to_char)
    for num in range(100000):
        chars = [ idx_to_char[ random.randint(0,charNum-1) ] for k in range(4) ]
        img = np.zeros((H,W,3),dtype=np.uint8) + 255
        print ''.join(chars)
        #dx = random.randint(-5,5)
        dx = 0
        cv2.putText( img, ' '.join(chars), (left+dx,bottom), font , 0.8,(0,0,0))
        cv2.imwrite(os.path.join(outdir,''.join(chars) + '.jpg'),img)
    
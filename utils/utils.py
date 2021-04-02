import sys
import numpy as np

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message) #print to screen
        self.log.write(message) #print to logfile

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def find_bb(volume):
    img_shape = volume.shape
    bb = np.zeros((6,), dtype=np.uint)
    bb_extend = 3
    # axis
    for i in range(img_shape[0]):
        img_slice_begin = volume[i,:,:]
        if np.sum(img_slice_begin)>0:
            bb[0] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[0]):
        img_slice_end = volume[img_shape[0]-1-i,:,:]
        if np.sum(img_slice_end)>0:
            bb[1] = np.min([img_shape[0]-1-i + bb_extend, img_shape[0]-1])
            break
    # seg
    for i in range(img_shape[1]):
        img_slice_begin = volume[:,i,:]
        if np.sum(img_slice_begin)>0:
            bb[2] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[1]):
        img_slice_end = volume[:,img_shape[1]-1-i,:]
        if np.sum(img_slice_end)>0:
            bb[3] = np.min([img_shape[1]-1-i + bb_extend, img_shape[1]-1])
            break

    # coronal
    for i in range(img_shape[2]):
        img_slice_begin = volume[:,:,i]
        if np.sum(img_slice_begin)>0:
            bb[4] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[2]):
        img_slice_end = volume[:,:,img_shape[2]-1-i]
        if np.sum(img_slice_end)>0:
            bb[5] = np.min([img_shape[2]-1-i+bb_extend, img_shape[2]-1])
            break
	
    return bb
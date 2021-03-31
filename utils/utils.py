import sys

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
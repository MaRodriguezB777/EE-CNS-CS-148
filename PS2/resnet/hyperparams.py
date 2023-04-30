
import numpy as np

class Resnet_Hyperparams():
    def __init__(self, train_loader):        
        self.iterations = len(train_loader)
        self.epochs =  None                                                # Authors cite 64k iterations
        self.lr = None                                                     # authors cite 0.1
        self.momentum = None                                               # authors cite 0.9
        self.weight_decay = None                                           # authors cite 0.0001
        self.milestones = [None, 
                           None]                                            # authors cite divide it by 10 at 32k and 48k iterations
        self.lr_multiplier = None                                           # ^
from helper import *
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from data_loader import *
from model import *


class InteractEOptimizer:

    def __init__(self
                 model: InteractE,
                 optimizer_name: str ="Adagrad"
                 batch_size: int = 256
                 decay1: float =)


    if (self.p.opt == 'adam'): 
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
    else:
        return torch.optim.SGD(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    
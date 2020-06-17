import tqdm
import torch
import numpy as np
from torch import nn
from torch import optim

from kelpie.models.tucker.model import TuckER # , KelpieTuckER
from kelpie.evaluation import Evaluator

class TuckEROptimizer:
    def __init__(self,
                 model: TuckER,
                 optimizer_name: str = "Adam",
                 scheduler_name: str = "ExponentialLR",
                 batch_size: int = 128,
                 learning_rate: float = 0.0005,
                 decay: float = 1.0,
                 label_smoothing: float = 0.1,
                 verbose: bool = True):
        self.model = model
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.verbose = verbose

        # build all the supported optimizers using the passed params (learning rate if Adam)
        supported_optimizers = {
            'Adam': optim.Adam(params=self.model.parameters(), lr=learning_rate)
        }

        # choose the Torch Optimizer object to use, based on the passed name
        self.optimizer = supported_optimizers[optimizer_name]
        
        # build all the supported schedulers using the passed params (decay if ExponentialLR)
        supported_schedulers = {
            'ExponentialLR': optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        }

        # create the evaluator to use between epochs
        self.evaluator = Evaluator(self.model)

    def train(self,
              train_samples: np.array,
              max_epochs: int,
              save_path: str = None,
              evaluate_every:int =-1,
              valid_samples:np.array = None):
        



    def epoch(self,
              batch_size: int,
              training_samples: np.array):


    def step_on_batch(self, loss, batch):
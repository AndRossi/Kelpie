import numpy
from link_prediction.evaluation.evaluation import Evaluator
from link_prediction.models.model import Model

class Optimizer:
    """
        The Optimizer class provides the interface that any LP Optimizer should implement.
    """

    def __init__(self,
                 model: Model,
                 hyperparameters: dict,
                 verbose: bool = True):

        self.model = model  #type:Model
        self.dataset = self.model.dataset
        self.verbose = verbose

        # create the evaluator to use between epochs
        self.evaluator = Evaluator(self.model)

    def train(self,
              train_samples: numpy.array,
              save_path: str = None,
              evaluate_every:int =-1,
              valid_samples:numpy.array = None):
        pass

    # def epoch(self,
    #          batch_size: int,
    #          training_samples: numpy.array):
    #    pass

    # def step_on_batch(self, loss, batch):
    #    pass



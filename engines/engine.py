import numpy

from dataset import Dataset
from model import Model

class ExplanationEngine:
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        self.model = model
        self.model.to('cuda')   # it this hasn't been done yet, load the model in GPU
        self.dataset = dataset
        self.hyperparameters = hyperparameters


    def simple_removal_explanations(self,
                                    sample_to_explain: numpy.array,
                                    perspective: str):
        pass
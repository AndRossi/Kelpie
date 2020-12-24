import itertools
from typing import Tuple, Any

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
                                    sample_to_explain: Tuple[Any, Any, Any],
                                    perspective: str,
                                    top_k: int):
        pass

    def simple_addition_explanations(self,
                    sample_to_explain: Tuple[Any, Any, Any],
                    perspective: str,
                    samples_to_add: list):
        pass

    def _extract_sample_nples(self, samples: list, n: int):
        return list(itertools.combinations(samples, n))
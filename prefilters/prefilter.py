from typing import Tuple, Any
from dataset import Dataset
from link_prediction.models.model import Model

class PreFilter:

    """
    The PreFilter object is the manager of the prefilter process.
    It implements the prefiltering pipeline.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset):
        """
        PreFilter constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        self.model = model
        self.dataset = dataset

    def top_promising_samples_for(self,
                                 sample_to_explain:Tuple[Any, Any, Any],
                                 perspective:str,
                                 top_k=50):

        """
        This method extracts the top k promising samples for interpreting the sample to explain,
        from the perspective of either its head or its tail (that is, either featuring its head or its tail).

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the explanation perspective. It can be either "head" or "tail":
                                - if "head", find the most promising samples featuring the head of the sample to explain
                                - if "tail", find the most promising samples featuring the tail of the sample to explain
        :param top_k: the number of top promising samples to extract.
        :return: the sorted list of the most promising samples.
        """
        pass
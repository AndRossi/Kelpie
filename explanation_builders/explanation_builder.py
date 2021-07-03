from typing import Tuple, Any
from dataset import Dataset
from link_prediction.models.model import Model

class SufficientExplanationBuilder:

    """
    The SufficientExplanationBuilder object guides the search for sufficient explanations.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str,
                 num_entities_to_convert=10):
        """
        SufficientRulesExtractor object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        self.model = model
        self.dataset = dataset
        self.sample_to_explain = sample_to_explain
        self.triple_to_explain = self.dataset.sample_to_fact(self.sample_to_explain)

        self.perspective = perspective
        self.perspective_entity = sample_to_explain[0] if perspective == "head" else sample_to_explain[2]

        self.num_entities_to_convert = num_entities_to_convert
        self.length_cap = 4

    def build_explanations(self,
                           samples_to_add: list,
                           top_k: int =10):
        pass

    def _average(self, l: list):
        result = 0.0
        for item in l:
            result += float(item)
        return result / float(len(l))


class NecessaryExplanationBuilder:


    """
    The NecessaryExplanationBuilder object guides the search for necessary explanations.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str):
        """
        NecessaryExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        self.model = model
        self.dataset = dataset
        self.sample_to_explain = sample_to_explain
        self.triple_to_explain = self.dataset.sample_to_fact(self.sample_to_explain)

        self.perspective = perspective
        self.perspective_entity = sample_to_explain[0] if perspective == "head" else sample_to_explain[2]

        self.length_cap = 4

    def build_explanations(self,
                           samples_to_add: list,
                           top_k: int =10):
        pass

    def _average(self, l:list):
        result = 0.0
        for item in l:
            result += float(item)
        return result/float(len(l))
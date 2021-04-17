from typing import Tuple, Any
from dataset import Dataset
from model import Model

class SufficientRuleExtractor:


    """
    The SufficientRuleExtractor object guides the search for sufficient rules.
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
        self.rule_length_cap = 4

    def extract_rules(self,
                      samples_to_add: list,
                      top_k: int =10):
        pass

    def _average(self, l: list):
        result = 0.0
        for item in l:
            result += float(item)
        return result / float(len(l))


class NecessaryRuleExtractor:


    """
    The NecessaryRuleExtractor object guides the search for necessary rules.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str):
        """
        NecessaryRuleExtractor object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        self.model = model
        self.dataset = dataset
        self.sample_to_explain = sample_to_explain
        self.triple_to_explain = self.dataset.sample_to_fact(self.sample_to_explain)

        self.perspective = perspective
        self.perspective_entity = sample_to_explain[0] if perspective == "head" else sample_to_explain[2]

        self.rule_length_cap = 4

    def extract_rules(self,
                      samples_to_add: list,
                      top_k: int =10):
        pass

    def _average(self, l:list):
        result = 0.0
        for item in l:
            result += float(item)
        return result/float(len(l))
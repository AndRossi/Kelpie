from typing import Tuple, Any
from dataset import Dataset
from model import Model

def average(l):
    s = 0.0
    for x in l:
        s += float(x)
    return s/float(len(l))


class SufficientRuleExtractor:


    """
    The SufficientRulesExtractor object guides the search for sufficient rules.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str,
                 num_entities_to_convert=10):
        """
        SufficientRulesExtractor object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
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

    def _avg(self, l: list):
        s = 0.0
        for x in l:
            s += float(x)
        return s / float(len(l))
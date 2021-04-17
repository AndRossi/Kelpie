import time
from typing import Tuple, Any
from dataset import Dataset
from engines.criage_engine import CriageEngine
from model import Model
from rule_extraction.rule_extractor import NecessaryRuleExtractor

class CriageNecessaryRuleExtractor(NecessaryRuleExtractor):

    """
    The CriageNecessaryRuleExtractor object guides the search for necessary facts to remove using the Criage approach
    """
    def __init__(self, model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str):
        """
        CriageNecessaryRuleExtractor object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param perspective
        """

        super().__init__(model, dataset, sample_to_explain, perspective)

        self.engine = CriageEngine(model=model,
                                   dataset=dataset,
                                   hyperparameters=hyperparameters)

    def extract_rules(self,
                      samples_to_remove: list,
                      top_k: int =10):

        rule_2_relevance = {}

        (head_to_explain, _, tail_to_explain) = self.sample_to_explain

        for i, sample_to_remove in enumerate(samples_to_remove):
            print("\n\tComputing relevance for sample " + str(i) + " on " + str(len(samples_to_remove)) + ": " +
                  self.dataset.printable_sample(sample_to_remove))

            tail_to_remove = sample_to_remove[2]

            if tail_to_remove == head_to_explain:
                perspective = "head"
            elif tail_to_remove == tail_to_explain:
                perspective = "tail"
            else:
                raise ValueError

            relevance = self.engine.removal_relevance(sample_to_explain=self.sample_to_explain,
                                                            perspective=perspective,
                                                            samples_to_remove=[sample_to_remove])

            rule_2_relevance[tuple([sample_to_remove])] = relevance

            cur_line = ";".join(self.triple_to_explain) + ";" + \
                        ";".join(self.dataset.sample_to_fact(sample_to_remove)) + ";" \
                       + str(relevance)

            with open("output_details_1.csv", "a") as output_file:
                output_file.writelines([cur_line + "\n"])


        # return sorted(rule_2_relevance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted(rule_2_relevance.items(), key=lambda x: x[1])[:top_k]

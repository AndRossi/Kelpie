from typing import Tuple, Any
from dataset import Dataset
from relevance_engines.data_poisoning_engine import DataPoisoningEngine
from link_prediction.models.model import Model, LEARNING_RATE
from explanation_builders.explanation_builder import NecessaryExplanationBuilder

class DataPoisoningNecessaryExplanationBuilder(NecessaryExplanationBuilder):

    """
    The DataPoisoningNecessaryExplanationBuilder object guides the search for DP necessary rules

    """
    def __init__(self, model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str):
        """
        DataPoisoningNecessaryExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param perspective
        """

        super().__init__(model, dataset, sample_to_explain, perspective)

        self.engine = DataPoisoningEngine(model=model,
                                          dataset=dataset,
                                          hyperparameters=hyperparameters,
                                          epsilon=hyperparameters[LEARNING_RATE])

    def build_explanations(self,
                           samples_to_remove: list,
                           top_k: int =10):
        rule_2_relevance = {}

        for i, sample_to_remove in enumerate(samples_to_remove):
            print("\n\tComputing relevance for sample " + str(i) + " on " + str(
                len(samples_to_remove)) + ": " + self.dataset.printable_sample(sample_to_remove))

            relevance, \
            original_target_entity_score, original_target_entity_rank, \
            original_removed_sample_score, perturbed_removed_sample_score \
                = self.engine.removal_relevance(sample_to_explain=self.sample_to_explain,
                                                            perspective=self.perspective,
                                                            samples_to_remove=[sample_to_remove])

            rule_2_relevance[tuple([sample_to_remove])] = relevance

            cur_line = ";".join(self.triple_to_explain) + ";" + \
                        ";".join(self.dataset.sample_to_fact(sample_to_remove)) + ";" + \
                        str(original_target_entity_score) + ";" + \
                        str(original_target_entity_rank) + ";" + \
                        str(original_removed_sample_score) + ";" + \
                        str(perturbed_removed_sample_score) + ";" + \
                        str(relevance)

            with open("output_details_1.csv", "a") as output_file:
                output_file.writelines([cur_line + "\n"])

        return sorted(rule_2_relevance.items(), key=lambda x: x[1], reverse=True)[:top_k]

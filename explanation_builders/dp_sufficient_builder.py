from collections import defaultdict
from typing import Tuple, Any
from dataset import Dataset
from relevance_engines.data_poisoning_engine import DataPoisoningEngine
from link_prediction.models.model import Model, LEARNING_RATE
from explanation_builders.explanation_builder import SufficientExplanationBuilder
import numpy

class DataPoisoningSufficientExplanationBuilder(SufficientExplanationBuilder):

    """
    The DataPoisoningSufficientExplanationBuilder object guides the search for sufficient explanations for DP
    """
    def __init__(self, model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str,
                 num_entities_to_convert=10,
                 entities_to_convert = None
                 ):
        """
        DataPoisoningSufficientExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param sample_to_explain
        :param perspective
        :param num_entities_to_convert
        """

        super().__init__(model, dataset, sample_to_explain, perspective, num_entities_to_convert)

        self.engine = DataPoisoningEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters,
                                         epsilon=hyperparameters[LEARNING_RATE])

        if entities_to_convert is not None:
            assert len(entities_to_convert) == num_entities_to_convert
            self.entities_to_convert = entities_to_convert
        else:
            self.entities_to_convert = self.engine.extract_entities_for(model=self.model,
                                                                        dataset=self.dataset,
                                                                        sample=sample_to_explain,
                                                                        perspective=perspective,
                                                                        k=num_entities_to_convert,
                                                                        degree_cap=200)
    def build_explanations(self,
                           samples_to_add: list,
                           top_k: int =10):


        rule_2_global_relevance = {}

        # this is an exception: all rules with length 1 are tested
        for i, sample_to_add in enumerate(samples_to_add):
            print("\n\tComputing relevance for sample " + str(i) + " on " + str(len(samples_to_add)) + ": " + self.dataset.printable_sample(sample_to_add))
            rule = tuple([sample_to_add])
            global_relevance = self._compute_relevance_for_rule(rule)
            rule_2_global_relevance[rule] = global_relevance
            print("\tObtained global relevance: " + str(global_relevance))

        return sorted(rule_2_global_relevance.items(), key=lambda x: x[1], reverse=True)[:top_k]


    def _compute_relevance_for_rule(self, rule: Tuple):
        rule_length = len(rule)
        assert(len(rule[0]) == 3)
        assert rule_length == 1

        sample_to_add = rule[0]

        rule_2_individual_relevances = defaultdict(lambda: [])
        outlines = []

        for j, entity_to_convert in enumerate(self.entities_to_convert):
            print("\t\tConverting entity " + str(j) + " on " + str(self.num_entities_to_convert) + ": " + self.dataset.entity_id_2_name[entity_to_convert])

            r_nple_to_add = Dataset.replace_entity_in_samples(samples=rule,
                                                              old_entity=self.perspective_entity,
                                                              new_entity=entity_to_convert,
                                                              as_numpy=False)
            r_sample_to_convert = Dataset.replace_entity_in_sample(self.sample_to_explain, self.perspective_entity, entity_to_convert)
            r_triple_to_convert = self.dataset.sample_to_fact(r_sample_to_convert)

            # if rule length is 1 try all r_samples_to_add and get their individual relevances
            individual_relevance, \
            original_target_entity_score, original_target_entity_rank, \
            original_added_sample_score, perturbed_added_sample_score = \
                self.engine.addition_relevance(sample_to_convert=r_sample_to_convert,
                                               perspective=self.perspective,
                                               samples_to_add=r_nple_to_add)

            rule_2_individual_relevances[rule].append(individual_relevance)


            outlines.append(";".join(self.triple_to_explain) + ";" + \
                            ";".join(r_triple_to_convert) + ";" + \
                            ";".join(self.dataset.sample_to_fact(sample_to_add)) + ";" + \
                            str(original_target_entity_score) + ";" + \
                            str(original_target_entity_rank) + ";" + \
                            str(original_added_sample_score) + ";" + \
                            str(perturbed_added_sample_score) + ";" + \
                            str(individual_relevance))

        # add the rule global relevance to all the outlines that refer to this rule
        global_relevance = self._average(rule_2_individual_relevances[rule])

        complete_outlines = [x + ";" + str(global_relevance) + "\n" for x in outlines]
        with open("output_details_" + str(rule_length) + ".csv", "a") as output_file:
            output_file.writelines(complete_outlines)

        return global_relevance

    def _preliminary_rule_score(self, rule, sample_2_relevance):
        return numpy.sum([sample_2_relevance[x] for x in rule])
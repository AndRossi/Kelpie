import itertools
from collections import defaultdict
from typing import Tuple, Any

import numpy

from dataset import Dataset
from engines.post_training_engine import PostTrainingEngine
from model import Model
from rule_extraction.rule_extractor import SufficientRuleExtractor

XSI_THRESHOLD = 0.9

class BruteForceSufficientRuleExtractor(SufficientRuleExtractor):

    """
    The BruteForceSufficientRuleExtractor object guides the search for sufficient rules with a brute force policy
    """
    def __init__(self, model: Model,
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

        super().__init__(model, dataset, hyperparameters, sample_to_explain, perspective, num_entities_to_convert)

        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)

        self.entities_to_convert = self.engine.extract_entities_for(model=self.model,
                                                                    dataset=self.dataset,
                                                                    sample=sample_to_explain,
                                                                    perspective=perspective,
                                                                    k=num_entities_to_convert,
                                                                    degree_cap=200)
    def extract_rules(self,
                      samples_to_add: list,
                      top_k: int =10):

        all_rules_with_relevance = []

        # get relevance for rules with length 1 (that is, samples)
        sample_2_relevance = self.extract_rules_with_length_1(samples_to_add=samples_to_add)

        samples_with_relevance = sorted(sample_2_relevance.items(), key=lambda x: x[1], reverse=True)
        all_rules_with_relevance += [([x], y) for (x, y) in samples_with_relevance]

        best_rule, best_rule_relevance = all_rules_with_relevance[0]
        if best_rule_relevance > XSI_THRESHOLD:
            return all_rules_with_relevance

        cur_rule_length = 2

        current_best_rule_relevance = -1
        while cur_rule_length <= self.rule_length_cap:
            rule_2_relevance = self.extract_rules_with_length(samples_to_add=samples_to_add,
                                                              length=cur_rule_length,
                                                              sample_2_relevance=sample_2_relevance)

            rules_with_relevance = sorted(rule_2_relevance.items(), key=lambda x: x[1], reverse=True)

            # this step is necessary because even one-sample-rules, that so far have been treated as simple samples,
            # must be returned as lists of samples to the calling method
            if cur_rule_length == 1:
                rules_with_relevance = [([x], y) for x, y in rules_with_relevance]

            best_rule, best_rule_relevance = rules_with_relevance[0]

            all_rules_with_relevance += rules_with_relevance

            if best_rule_relevance > current_best_rule_relevance:
                current_best_rule = best_rule
                current_best_rule_relevance = best_rule_relevance
            # else:
            #   break       if searching for additional rules does not seem promising, you should exit

            if current_best_rule_relevance > XSI_THRESHOLD:
                break

            cur_rule_length += 1

        return sorted(all_rules_with_relevance, key=lambda x: x[1], reverse=True)[:top_k]


    def extract_rules_with_length(self,
                                  samples_to_add: list,
                                  length: int,
                                  sample_2_relevance: dict):

        if length == 1:
            return self.extract_rules_with_length_1(samples_to_add=samples_to_add)

        rule_2_global_relevance = {}

        all_possible_rules = itertools.combinations(samples_to_add, length)
        all_possible_rules_with_preliminary_scores = [(x, self._preliminary_rule_score(x, sample_2_relevance)) for x in all_possible_rules]
        all_possible_rules_with_preliminary_scores = sorted(all_possible_rules_with_preliminary_scores, key=lambda x:x[1], reverse=True)


        for i in range(len(all_possible_rules_with_preliminary_scores)):

            current_rule, current_preliminary_score = all_possible_rules_with_preliminary_scores[i]

            print("\n\tComputing relevance for rule " + str(i) + " on " + str(len(all_possible_rules_with_preliminary_scores)) + ": " + self.dataset.printable_nple(current_rule))
            relevance = self._compute_relevance_for_rule(current_rule)
            print("\tGlobal relevance for this rule across all entities to convert: " + str(relevance))
            rule_2_global_relevance[current_rule] = relevance

            if relevance > XSI_THRESHOLD:
                break

        return rule_2_global_relevance


    def extract_rules_with_length_1(self,
                                    samples_to_add: list):

        rule_2_global_relevance = {}
        for i, sample_to_add in enumerate(samples_to_add):
            print("\n\tComputing relevance for sample " + str(i) + " on " + str(len(samples_to_add)) + ": " + self.dataset.printable_sample(sample_to_add))
            relevance = self._compute_relevance_for_rule(tuple([sample_to_add]))
            rule_2_global_relevance[sample_to_add] = relevance

        return rule_2_global_relevance


    def _compute_relevance_for_rule(self, rule: Tuple):

        rule_length = len(rule)
        assert(len(rule[0]) == 3)

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
            original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
            base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank, \
            pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank = \
                self.engine.addition_relevance(sample_to_convert=r_sample_to_convert,
                                                      perspective=self.perspective,
                                                      samples_to_add=r_nple_to_add)

            rule_2_individual_relevances[rule].append(individual_relevance)

            outlines.append(";".join(self.triple_to_explain) + ";" + \
                            ";".join(r_triple_to_convert) + ";" + \
                            ";".join([";".join(self.dataset.sample_to_fact(x)) for x in r_nple_to_add]) + ";" + \
                            str(original_best_entity_score) + ";" + \
                            str(original_target_entity_score) + ";" + \
                            str(original_target_entity_rank) + ";" + \
                            str(base_pt_best_entity_score) + ";" + \
                            str(base_pt_target_entity_score) + ";" + \
                            str(base_pt_target_entity_rank) + ";" + \
                            str(pt_best_entity_score) + ";" + \
                            str(pt_target_entity_score) + ";" + \
                            str(pt_target_entity_rank) + ";" + \
                            str(individual_relevance))

        # add the rule global relevance to all the outlines that refer to this rule
        global_relevance = self._avg(rule_2_individual_relevances[rule])

        complete_outlines = [x + ";" + str(global_relevance) + "\n" for x in outlines]
        with open("output_details_" + str(rule_length) + ".csv", "a") as output_file:
            output_file.writelines(complete_outlines)

        return global_relevance

    def _preliminary_rule_score(self, rule, sample_2_relevance):
        return numpy.sum([sample_2_relevance[x] for x in rule])

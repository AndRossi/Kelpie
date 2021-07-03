import itertools
import random
from collections import defaultdict
from typing import Tuple, Any

from dataset import Dataset
from relevance_engines.post_training_engine import PostTrainingEngine
from link_prediction.models.model import Model
from explanation_builders.explanation_builder import SufficientExplanationBuilder
import numpy

XSI_THRESHOLD = 0.9

class StochasticSufficientExplanationBuilder(SufficientExplanationBuilder):

    """
    The StochasticSufficientExplanationBuilder object guides the search for sufficient explanations with a probabilistic policy
    """
    def __init__(self, model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str,
                 num_entities_to_convert=10,
                 entities_to_convert=None):
        """
        StochasticSufficientExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param sample_to_explain
        :param perspective
        :param num_entities_to_convert
        """

        super().__init__(model, dataset, sample_to_explain, perspective, num_entities_to_convert)

        self.window_size = 10

        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)

        if entities_to_convert is not None:
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


        all_rules_with_relevance = []

        # get relevance for rules with length 1 (that is, samples)
        sample_2_relevance = self.extract_rules_with_length_1(samples_to_add=samples_to_add)

        samples_with_relevance = sorted(sample_2_relevance.items(), key=lambda x: x[1], reverse=True)
        samples_number = len(samples_with_relevance)
        all_rules_with_relevance += [([x], y) for (x, y) in samples_with_relevance]

        best_rule, best_rule_relevance = all_rules_with_relevance[0]
        if best_rule_relevance > XSI_THRESHOLD:
            return all_rules_with_relevance

        cur_rule_length = 2
        # stop if you have too few samples (e.g. if you have only 2 samples, you can not extract rules of length 3)
        # or if you get to the length cap
        while cur_rule_length <= samples_number and cur_rule_length <= self.length_cap:
            rule_2_relevance = self.extract_rules_with_length(samples_to_add=samples_to_add,
                                                              length=cur_rule_length,
                                                              sample_2_relevance=sample_2_relevance)
            current_rules_with_relevance = sorted(rule_2_relevance.items(), key=lambda x: x[1], reverse=True)

            all_rules_with_relevance += current_rules_with_relevance

            current_best_rule, current_best_rule_relevance = current_rules_with_relevance[0]

            if current_best_rule_relevance > best_rule_relevance:
                best_rule, best_rule_relevance = current_best_rule, current_best_rule_relevance
            # else:
            #   break       if searching for additional rules does not seem promising, you should exit now

            if best_rule_relevance > XSI_THRESHOLD:
                break

            cur_rule_length += 1

        return sorted(all_rules_with_relevance, key=lambda x: x[1], reverse=True)[:top_k]

    def extract_rules_with_length_1(self,
                                    samples_to_add: list):

        rule_2_global_relevance = {}

        # this is an exception: all rules with length 1 are tested
        for i, sample_to_add in enumerate(samples_to_add):
            print("\n\tComputing relevance for sample " + str(i) + " on " + str(len(samples_to_add)) + ": " + self.dataset.printable_sample(sample_to_add))
            global_relevance = self._compute_relevance_for_rule(tuple([sample_to_add]))
            rule_2_global_relevance[sample_to_add] = global_relevance
            print("\tObtained global relevance: " + str(global_relevance))

        return rule_2_global_relevance

    def extract_rules_with_length(self,
                                  samples_to_add: list,
                                  length: int,
                                  sample_2_relevance: dict):

        all_possible_rules = itertools.combinations(samples_to_add, length)
        all_possible_rules_with_preliminary_scores = [(x, self._preliminary_rule_score(x, sample_2_relevance)) for x in all_possible_rules]
        all_possible_rules_with_preliminary_scores = sorted(all_possible_rules_with_preliminary_scores, key=lambda x:x[1], reverse=True)

        rule_2_relevance = {}

        terminate = False
        best_relevance_so_far = -1e6    # initialize with an absurdly low value

        # initialize the relevance window with the proper size
        sliding_window = [None for _ in range(self.window_size)]

        i = 0
        while i < len(all_possible_rules_with_preliminary_scores) and not terminate:

            current_rule, current_preliminary_score = all_possible_rules_with_preliminary_scores[i]

            print("\n\tComputing relevance for rule: " + self.dataset.printable_nple(current_rule))
            current_rule_relevance = self._compute_relevance_for_rule(current_rule)
            rule_2_relevance[current_rule] = current_rule_relevance
            print("\n\tObtained global relevance: " + str(current_rule_relevance))

            # put the obtained relevance in the window
            sliding_window[i % self.window_size] = current_rule_relevance

            # early termination
            if current_rule_relevance > XSI_THRESHOLD:
                i += 1
                return rule_2_relevance

            # else, if the current relevance value is an improvement over the best relevance value seen so far, continue
            elif best_relevance_so_far is None or current_rule_relevance >= best_relevance_so_far:
                best_relevance_so_far = current_rule_relevance
                i += 1
                continue

            # else, if the window has not been filled yet, continue
            elif i < self.window_size:
                i += 1
                continue

            # else, use the average of the relevances in the window to assess the termination condition
            else:
                cur_avg_window_relevance = self._average(sliding_window)
                terminate_threshold = cur_avg_window_relevance/best_relevance_so_far
                random_value = random.random()
                terminate = random_value > terminate_threshold               # termination condition

                print("\n\tCurrent relevance " + str(current_rule_relevance))
                print("\tCurrent averaged window relevance " + str(cur_avg_window_relevance))
                print("\tMax relevance seen so far " + str(best_relevance_so_far))
                print("\tTerminate threshold:" + str(terminate_threshold))
                print("\tRandom value:" + str(random_value))
                print("\tTerminate:" + str(terminate))
                i+= 1

        return rule_2_relevance

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
            pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, \
            execution_time = self.engine.addition_relevance(sample_to_convert=r_sample_to_convert,
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
                            str(execution_time) + ";" + \
                            str(individual_relevance))

        # add the rule global relevance to all the outlines that refer to this rule
        global_relevance = self._average(rule_2_individual_relevances[rule])

        complete_outlines = [x + ";" + str(global_relevance) + "\n" for x in outlines]
        with open("output_details_" + str(rule_length) + ".csv", "a") as output_file:
            output_file.writelines(complete_outlines)

        return global_relevance

    def _preliminary_rule_score(self, rule, sample_2_relevance):
        return numpy.sum([sample_2_relevance[x] for x in rule])
import itertools
import random
from typing import Tuple, Any

from dataset import Dataset
from relevance_engines.post_training_engine import PostTrainingEngine
from link_prediction.models.model import *
from explanation_builders.explanation_builder import NecessaryExplanationBuilder
import numpy
import os
from collections import defaultdict

DEAFAULT_XSI_THRESHOLD = 5


class StochasticNecessaryExplanationBuilder(NecessaryExplanationBuilder):
    """
    The StochasticNecessaryExplanationBuilder object guides the search for necessary rules with a probabilistic policy
    """

    def __init__(self, model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str,
                 relevance_threshold: float = None,
                 max_explanation_length: int = -1):
        """
        StochasticSufficientExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param sample_to_explain: the predicted sample to explain
        :param perspective: the explanation perspective, either "head" or "tail"
        :param max_explanation_length: the maximum number of facts to include in the explanation to extract
        """

        super().__init__(model=model, dataset=dataset,
                         sample_to_explain=sample_to_explain, perspective=perspective,
                         max_explanation_length=max_explanation_length)

        self.args = dataset.args
        self.xsi = relevance_threshold if relevance_threshold is not None else DEAFAULT_XSI_THRESHOLD
        self.window_size = 10
        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)


    def prefilter_negative(self, all_rules, top_k):
        for i in range(0, top_k):
            if all_rules[i][1] < 0:
                break
        print(f'select top rules: {i}/{len(all_rules)+1}')
        return all_rules[:i]


    def build_explanations(self,
                           samples_to_remove: list,
                           top_k: int = 10):

        all_rules_with_relevance = []

        print(f'all possible rules with length 1:', len(samples_to_remove))
        # get relevance for rules with length 1 (that is, samples)
        sample_2_relevance = self.extract_rules_with_length_1(samples_to_remove=samples_to_remove)
        
        samples_with_relevance = sorted(sample_2_relevance.items(), key=lambda x: x[1], reverse=True)
        all_rules_with_relevance += [([x], y) for (x, y) in samples_with_relevance]
        
        samples_number = len(samples_with_relevance)

        best_rule, best_rule_relevance = all_rules_with_relevance[0]
        if best_rule_relevance > self.xsi:
            print('\tEarly termination after length 1')
            return self.prefilter_negative(all_rules_with_relevance, top_k)

        cur_rule_length = 2

        # stop if you have too few samples (e.g. if you have only 2 samples, you can not extract rules of length 3)
        # or if you get to the length cap
        while cur_rule_length <= samples_number and cur_rule_length <= self.length_cap:
            rule_2_relevance = self.extract_rules_with_length(samples_to_remove=samples_to_remove,
                                                              length=cur_rule_length,
                                                              sample_2_relevance=sample_2_relevance)
            current_rules_with_relevance = sorted(rule_2_relevance.items(), key=lambda x: x[1], reverse=True)

            all_rules_with_relevance += current_rules_with_relevance
            current_best_rule, current_best_rule_relevance = current_rules_with_relevance[0]

            if current_best_rule_relevance > best_rule_relevance:
                best_rule, best_rule_relevance = current_best_rule, current_best_rule_relevance
            # else:
            #   break       if searching for additional rules does not seem promising, you should exit now

            if best_rule_relevance > self.xsi:
                break

            cur_rule_length += 1

        # 只去 > 0 的前 k 个
        all_rules = sorted(all_rules_with_relevance, key=lambda x: x[1], reverse=True)
        return self.prefilter_negative(all_rules, top_k)



    def extract_rules_with_length_1(self, samples_to_remove: list):

        sample_2_relevance = {}

        # this is an exception: all samples (= rules with length 1) are tested
        for i, sample_to_remove in enumerate(samples_to_remove):
            relevance = self._compute_relevance_for_rule(([sample_to_remove]))
            sample_2_relevance[sample_to_remove] = relevance
            print(f"\t{i+1}/{len(samples_to_remove)}: {self.dataset.printable_sample(sample_to_remove)} {relevance}")
        return sample_2_relevance

    def extract_rules_with_length(self,
                                  samples_to_remove: list,
                                  length: int,
                                  sample_2_relevance: dict):

        all_possible_rules = itertools.combinations(samples_to_remove, length)
        all_possible_rules_with_preliminary_scores = [(x, self._preliminary_rule_score(x, sample_2_relevance)) for x in
                                                      all_possible_rules]
        all_possible_rules_with_preliminary_scores = sorted(all_possible_rules_with_preliminary_scores,
                                                            key=lambda x: x[1], reverse=True)

        rule_2_relevance = {}

        terminate = False
        best_relevance_so_far = -1e6  # initialize with an absurdly low value

        # initialize the relevance window with the proper size
        sliding_window = [None for _ in range(self.window_size)]

        i = 0
        print(f'all possible rules with length {length}:', len(all_possible_rules_with_preliminary_scores))
        while i < len(all_possible_rules_with_preliminary_scores) and not terminate:

            current_rule, current_preliminary_score = all_possible_rules_with_preliminary_scores[i]

            current_rule_relevance = self._compute_relevance_for_rule(current_rule)
            rule_2_relevance[current_rule] = current_rule_relevance
            print(f"\trule: {self.dataset.printable_nple(current_rule)} {current_rule_relevance}")

            # put the obtained relevance in the window
            sliding_window[i % self.window_size] = current_rule_relevance

            # early termination
            if current_rule_relevance > self.xsi:
                i += 1
                print(f"\tEarly Terminate: {current_rule_relevance} > {self.xsi}")
                terminate_at(length, i)
                return rule_2_relevance

            # else, if the current relevance value is an improvement over the best relevance value seen so far, continue
            elif current_rule_relevance >= best_relevance_so_far:
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
                terminate_threshold = cur_avg_window_relevance / best_relevance_so_far
                random_value = random.random()
                terminate = random_value > terminate_threshold  # termination condition
                i += 1

                def print_relevance(dic):
                    for k, v in dic.items():
                        print(f"{k}: {round(v, 4)}", end=', ')

                print('\t\t', end='')
                print_relevance({
                    "Current": current_rule_relevance,
                    "Current window": cur_avg_window_relevance, 
                    "Max": best_relevance_so_far,
                    "Terminate threshhold": terminate_threshold
                })
                
                if terminate:
                    print("Terminate!")
                    self.terminate_at(length, i)
                else:
                    print()

        return rule_2_relevance

    def _compute_relevance_for_rule(self, nple_to_remove: list):
        rule_length = len(nple_to_remove)

        # convert the nple to remove into a list
        assert (len(nple_to_remove[0]) == 3)

        relevance, \
        original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
        base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank, \
        pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, execution_time = \
            self.engine.removal_relevance(sample_to_explain=self.sample_to_explain,
                                          perspective=self.perspective,
                                          samples_to_remove=nple_to_remove)

        cur_line = ";".join(self.triple_to_explain) + ";" + \
                   ";".join([";".join(self.dataset.sample_to_fact(x)) for x in nple_to_remove]) + ";" + \
                    str(original_best_entity_score) + ";" + \
                    str(original_target_entity_score) + ";" + \
                    str(original_target_entity_rank) + ";" + \
                    str(base_pt_best_entity_score) + ";" + \
                    str(base_pt_target_entity_score) + ";" + \
                    str(base_pt_target_entity_rank) + ";" + \
                    str(pt_best_entity_score) + ";" + \
                    str(pt_target_entity_score) + ";" + \
                    str(pt_target_entity_rank) + ";" + \
                    str(relevance) + ";" + \
                    str(execution_time)

        with open(os.path.join(self.args.output_folder, "output_details_" + str(rule_length) + ".csv"), "a") as output_file:
            output_file.writelines([cur_line + "\n"])

        return relevance

    def _preliminary_rule_score(self, rule, sample_2_relevance):
        return numpy.sum([sample_2_relevance[x] for x in rule])

    def _average(self, l: list):
        result = 0.0
        for item in l:
            result += float(item)
        return result / float(len(l))

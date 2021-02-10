from collections import defaultdict
from typing import Tuple, Any

from dataset import Dataset
from engines.post_training_engine import PostTrainingEngine
from model import Model


class SufficientRulesExtractor:

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

        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)

        self.sample_to_explain = sample_to_explain
        self.triple_to_explain = self.dataset.sample_to_fact(self.sample_to_explain)

        self.perspective = perspective
        self.perspective_entity = sample_to_explain[0] if perspective == "head" else sample_to_explain[2]

        self.num_entities_to_convert = num_entities_to_convert
        self.entities_to_convert = self.engine.extract_entities_for(model=self.model,
                                                                    dataset=self.dataset,
                                                                    sample=sample_to_explain,
                                                                    perspective=perspective,
                                                                    k=num_entities_to_convert,
                                                                    degree_cap=200)

    def extract_rules(self,
                     samples_to_add: list,
                     rule_length=1):

        outlines = []
        rule_2_coverage = defaultdict(lambda: 0)

        # for each entity to convert:
        #   - replace the perspective entity with the entity to convert in the sample to explain
        #   - replace the perspective entity with the entity to convert in all the samples to add
        # We call the samples in which the perspective entity has been replaced with entity to convert "r_samples"
        #
        # Then, obtain the individual relevance in addition for each sample in r_samples_to_add
        # (with respect to the conversion r_sample_to_convert)
        for i, entity_to_convert in enumerate(self.entities_to_convert):
            print("\n\tConverting entity " + str(i) + " on " + str(self.num_entities_to_convert) + ": " + self.dataset.entity_id_2_name[entity_to_convert])

            r_sample_to_convert = Dataset.replace_entity_in_sample(self.sample_to_explain, self.perspective_entity, entity_to_convert)
            r_triple_to_convert = self.dataset.sample_to_fact(r_sample_to_convert)

            r_samples_to_add = Dataset.replace_entity_in_samples(samples=samples_to_add,
                                                                 old_entity=self.perspective_entity,
                                                                 new_entity=entity_to_convert,
                                                                 as_numpy=False)

            # if rule length is 1 try all r_samples_to_add and get their individual relevances
            if rule_length == 1:
                r_samples_with_relevance, \
                r_sample_2_detailed_results, \
                original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
                base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank = \
                    self.engine.simple_addition_explanations(sample_to_convert=r_sample_to_convert,
                                                             perspective=self.perspective,
                                                             samples_to_add=r_samples_to_add)

                for r_added_sample in r_sample_2_detailed_results:
                    r_added_triple = self.dataset.sample_to_fact(r_added_sample)
                    pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank = r_sample_2_detailed_results[r_added_sample]

                    outlines.append(";".join(self.triple_to_explain) + ";" + \
                                           ";".join(r_triple_to_convert) + ";" + \
                                           ";".join(r_added_triple) + ";" + \
                                            str(original_best_entity_score) + ";" + \
                                            str(original_target_entity_score) + ";" + \
                                            str(original_target_entity_rank) + ";" + \
                                            str(base_pt_best_entity_score) + ";" + \
                                            str(base_pt_target_entity_score) + ";" + \
                                            str(base_pt_target_entity_rank) + ";" + \
                                            str(pt_best_entity_score) + ";" + \
                                            str(pt_target_entity_score) + ";" + \
                                            str(pt_target_entity_rank) + "\n")

                # for each sample that we have tried to add (and for which we have computed relevance)
                # go back to the sample containing the perspective entity
                # and update its coverage
                for j, cur_r_sample_with_relevance in enumerate(r_samples_with_relevance):

                    cur_r_sample, cur_relevance = cur_r_sample_with_relevance
                    cur_sample = Dataset.replace_entity_in_sample(sample=cur_r_sample,
                                                                  old_entity=entity_to_convert,
                                                                  new_entity=self.perspective_entity,
                                                                  as_numpy=False)

                    rule_2_coverage[cur_sample] += 1.0/(float(j+1.0)*self.num_entities_to_convert)


        rules_with_coverage = sorted(rule_2_coverage.items(), key=lambda x:x[1], reverse=True)[:10]

        with open("output_details_1.csv", "a") as output_simple:
            output_simple.writelines(outlines)

        return rules_with_coverage
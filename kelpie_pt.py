from collections import defaultdict
from typing import Tuple, Any

from dataset import Dataset
from engines.post_training_engine import PostTrainingEngine
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.optimization.bce_optimizer import KelpieBCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import KelpieMultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import KelpiePairwiseRankingOptimizer
from model import Model


class Kelpie:
    """
    The Kelpie object is the overall manager of the explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations to the ExplanationEngines
    and to the entity_similarity modules.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        """
        Kelpie object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        """
        self.model = model
        self.dataset = dataset

        #self.engine = GradientEngine(model=model,
        #                             dataset=dataset,
        #                             hyperparameters=hyperparameters,
        #                             epsilon=hyperparameters[LEARNING_RATE])

        if isinstance(self.model, ComplEx):
            optimizer_class = KelpieMultiClassNLLOptimizer
        elif isinstance(self.model, ConvE):
            optimizer_class = KelpieBCEOptimizer
        elif isinstance(self.model, TransE):
            optimizer_class = KelpiePairwiseRankingOptimizer
        else:
            optimizer_class = KelpieMultiClassNLLOptimizer

        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters,
                                         post_training_optimizer_class=optimizer_class)

    def explain_old(self,
                sample_to_explain:Tuple[Any, Any, Any],
                perspective:str,
                num_similar_entities=10,
                num_relevant_samples=50):
        """
        This method extracts explanations for a specific sample, from the perspective of either its head or its taiL,

        The explanation pipeline is the following:
            1. extract the top relevant training samples for the sample_to_explain
            2. identify a set of specimen entities, based on their similarity to
               either the sample head or the sample tail (depending on the chosen perspective),
               and for which the model currently does not predict the same prediction as the sample to explain.
            3. identify which of the training samples extracted in phase 1, if added to the entities extracted in phase 2,
               can boost their prediction making them more similar to the prediction of the sample to explain.
            4. identify which combinations of the training samples extracted in phase 1,
               if added to the entities extracted in phase 2,
               can boost their prediction making them more similar to the prediction of the sample to explain.

        For instance, if the sample to explain is <Barack Obama, nationality, USA> and the perspective is "head",
        Kelpie finds explanations to the question "Given Barack Obama and nationality, why does the model predict USA?":
            1. it extracts the training samples featuring Barack Obama
               that are most relevant to <Barack Obama, nationality, USA>;
            2. it extracts a set of specimen entities "e" based on their similarity to Barack Obama,
               conditioned to relation nationality and tail USA,
               and for which <"e", nationality, ?> does not rank "USA" as tail in first position (filtered scenario)
            3. it identifies which of the Barack Obama samples extracted in phase 1,
               if added to the entities "e" extracted in phase 2, can boost the <"e", nationality, USA> prediction
            4. identify which combinations of the of the Barack Obama samples extracted in 1,
               if added to the "e" entities extracted in phase 2, can boost the <"e", nationality, USA> prediction.

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the perspective of the requested explanations.
                            It can be either "head" or "tail":
                                - if "head", Kelpie answers the question
                                    "given the sample head and relation, why is the sample tail predicted as tail?"
                                - if "tail", Kelpie answers the question
                                    "given the sample relation and tail, why is the sample head predicted as head?"
        :param num_similar_entities: the number of entities that must be extracted
                                     based on their similarity to the perspective entity
        :param num_relevant_samples: the number of samples relevant to the sample to explain
                                     that must be identified and added to the extracted similar entities
                                     to verify whether they boost the target prediction or not
        :return: two lists:
                    the first one contains, for each relevant sample, a couple containing the sample and an index of its global relevance across the similar entities
                    the second one contains, for each combination of relevant samples, a couple containing the combination and its relevance across the similar entities

        """

        outlines_simple = []
        outlines_comb = []

        # identify the perspective entity entity in the sample to explain
        head, relation, tail = sample_to_explain
        perspective_entity = head if perspective == "head" else tail

        # extract the training samples featuring the perspective entity and most relevant to the sample to explain
        train_samples_with_relevance = self.engine.simple_removal_explanations(sample_to_explain=sample_to_explain,
                                                                               perspective=perspective,
                                                                               top_k=num_relevant_samples)

        print("\tRemoval Relevances: ")
        for x in train_samples_with_relevance:
            print("\t\t" + ";".join(self.dataset.sample_to_fact(x[0])) + ": " + str(x[1]))
        print()
        most_relevant_train_samples = [x[0] for x in train_samples_with_relevance[:num_relevant_samples]]

        # extract the top entities most "comparable" to the perspective entity
        comparable_entities = self.engine.extract_entities_for(model=self.model,
                                                               dataset=self.dataset,
                                                               sample=sample_to_explain,
                                                               perspective=perspective,
                                                               k=num_similar_entities,
                                                               degree_cap=200)

        original_sample_2_coverage = defaultdict(lambda: 0.0)
        original_nple_2_coverage = defaultdict(lambda: 0.0)

        # for each comparable entity
        # add each of the most_relevant_train_samples to the comparable entity
        # and individually check its relevance

        for k, comparable_entity in enumerate(comparable_entities):
            print("\n\tConverting entity " + str(k) + " on " + str(len(comparable_entities)) + ": " + self.dataset.entity_id_2_name[comparable_entity])
            # replace the perspective entity with the comparable_entity
            # both in the sample to explain and in all the samples to add
            comparable_sample_to_explain = Dataset.replace_entity_in_sample(sample_to_explain, perspective_entity, comparable_entity)

            # replace the perspective entity with the comparable_entity
            comparable_samples_to_add = Dataset.replace_entity_in_samples(samples=most_relevant_train_samples,
                                                                          old_entity=perspective_entity,
                                                                          new_entity=comparable_entity,
                                                                          as_numpy=False)

            # obtain the relevance of those samples in addition, with respect to the the sample to explain
            comparable_samples_with_relevance, \
            comparable_sample_2_pt_results, \
            original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
            base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank = \
                self.engine.simple_addition_explanations(sample_to_convert=comparable_sample_to_explain,
                                                         perspective=perspective,
                                                         samples_to_add=comparable_samples_to_add)

            for added_comparable_sample in comparable_sample_2_pt_results:
                pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank = comparable_sample_2_pt_results[added_comparable_sample]

                original_triple_to_convert = self.dataset.sample_to_fact(sample_to_explain)
                comparable_triple_to_convert = self.dataset.sample_to_fact(comparable_sample_to_explain)
                added_comparable_triple = self.dataset.sample_to_fact(added_comparable_sample)

                outlines_simple.append(";".join(original_triple_to_convert) + ";" + \
                                       ";".join(comparable_triple_to_convert) + ";" + \
                                       ";".join(added_comparable_triple) + ";" + \
                                        str(original_best_entity_score) + ";" + \
                                        str(original_target_entity_score) + ";" + \
                                        str(original_target_entity_rank) + ";" + \
                                        str(base_pt_best_entity_score) + ";" + \
                                        str(base_pt_target_entity_score) + ";" + \
                                        str(base_pt_target_entity_rank) + ";" + \
                                        str(pt_best_entity_score) + ";" + \
                                        str(pt_target_entity_score) + ";" + \
                                        str(pt_target_entity_rank) + "\n")


            for i, cur_comparable_sample_with_relevance in enumerate(comparable_samples_with_relevance):
                # for each cur_comparable_sample

                cur_comparable_sample, cur_relevance = cur_comparable_sample_with_relevance
                # go back to the sample containing the perspective entity
                cur_sample = Dataset.replace_entity_in_sample(sample=cur_comparable_sample,
                                                              old_entity=comparable_entity,
                                                              new_entity=perspective_entity,
                                                              as_numpy=False)

                original_sample_2_coverage[cur_sample] += 1.0/(float(i+1.0)*len(comparable_entities))


            #out = self.engine.nple_addition_explanations(sample_to_explain=comparable_sample_to_explain,
            #                                             perspective=perspective,
            #                                             samples_to_add=comparable_samples_to_add,
            #                                             n=2)
            #if len(out) == 4:
            #    comparable_nples_with_relevance, \
            #    comparable_nple_2_perturbed_score, \
            #    target_entity_score, best_entity_score = out
            #else:
            #    continue

            #for comparable_nple in comparable_nple_2_perturbed_score:
            #    outlines_comb.append("+".join([";".join(self.dataset.sample_to_fact(x)) for x in comparable_nple]) + ";" + \
            #                         str(best_entity_score) + ";" + \
            #                         str(target_entity_score) + ";" + \
            #                         str(comparable_nple_2_perturbed_score[comparable_nple]) + "\n")
            outlines_comb.append("\n")
            #for i, cur_comparable_nple_with_relevance in enumerate(comparable_nples_with_relevance):
            #    cur_comparable_nple = cur_comparable_nple_with_relevance[0]
            #    cur_nple = tuple([Dataset.replace_entity_in_sample(sample=cur_sample,
            #                                                       old_entity=comparable_entity,
            #                                                       new_entity=perspective_entity,
            #                                                       as_numpy=False) for cur_sample in cur_comparable_nple])
            #    original_nple_2_coverage[cur_nple] += 1.0/(float(i+1.0)*len(comparable_entities))

        expl_nples = []
        expl_samples = sorted(original_sample_2_coverage.items(), key=lambda x:x[1], reverse=True)[:10]
        #expl_nples = sorted(original_nple_2_coverage.items(), key=lambda x:x[1], reverse=True)[:10]

        with open("output_details_1.csv", "a") as output_simple:
            output_simple.writelines(outlines_simple)

        with open("output_details_comb.csv", "a") as output_comb:
            output_comb.writelines(outlines_comb)

        return expl_samples, expl_nples, comparable_entities
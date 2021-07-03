import itertools
import random
from typing import Tuple, Any

import numpy
import torch

from dataset import Dataset, ONE_TO_ONE, MANY_TO_ONE
from link_prediction.models.model import Model


class ExplanationEngine:
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        self.model = model
        self.model.to('cuda')   # it this hasn't been done yet, load the model in GPU
        self.model.eval()
        self.dataset = dataset
        self.hyperparameters = hyperparameters


    def simple_removal_explanations(self,
                                    sample_to_explain: Tuple[Any, Any, Any],
                                    perspective: str,
                                    top_k: int):
        pass

    def simple_addition_explanations(self,
                    sample_to_explain: Tuple[Any, Any, Any],
                    perspective: str,
                    samples_to_add: list):
        pass

    def _extract_sample_nples(self, samples: list, n: int):
        return list(itertools.combinations(samples, n))


    def extract_entities_for(self,
                             model: Model,
                             dataset: Dataset,
                             sample: numpy.array,
                             perspective: str,
                             k: int,
                             degree_cap=-1):
        """
            Extract k entities to replace the perspective entity in the passed sample.

            The purpose of such entities is to allow the engine to identify sufficient rules to explain the sample.
            To do so, the engine replaces the perspective entity in the sample with the extracted entities,
            and the engine analyzes the effect of adding/removing fact featuring those entities.

            The whole system works assuming that the extracted entities, when put in the passed sample,
            result in *wrong* facts, that are not predicted as true by the model;
            the purpose of the engine is identify the minimal combination of facts to added to those entities
            in order to "fool" the model and make it predict those "wrong" facts as true.

            As a consequence the extracted entities will adhere to the following criteria:
                - must be different from the perspective entity of the sample to explain (obviously)
                - it must be seen in training (obviously)
                - the extracted entity must form a "true" fact when put in the sample.
                  E.g., Michelle can not be a good replacement for Barack in <Barack, parent, Natasha>
                  if <Michelle, parent, Natasha> is already present in the dataset (in train, valid or test set)
                - if the relation in the sample has *_TO_ONE type, the extracted entities must not already have
                  a known tail for the relation under analysis in the training set.
                  (e.g when explaining <Barack, nationality, USA> we can not use entity "Vladimir" to replace Barack
                  if <Vladimir, nationality, Russia> is either in train, valid or test set).
                - the extracted entity must not form a "true" fact when put in the sample.
                  E.g., Michelle can not be a good replacement for Barack in <Barack, parent, Natasha>
                  if <Michelle, parent, Natasha> is already present in the dataset (in train, valid or test set)
                - the extracted entity must not form a fact that is predicted by the model.
                  (we want current_other_entities that, without additions, do not predict the target entity with rank 1)
                  (e.g. if we are explaining <Barack, nationality, USA>, George is an acceptable "convertible entity"
                  only if <George, nationality, ?> does not already rank USA in 1st position!)

            :param model: the model the prediction of which must be explained
            :param dataset: the dataset on which the passed model has been trained
            :param sample: the sample that the engine is trying to explain
            :param perspective: the perspective from which to explain the passed sample: either "head" or "tail".
            :param k: the number of entities to extract
            :param degree_cap:
            :return:
        """

        # this is EXTREMELY important in models with dropout and/or batch normalization.
        # It basically disables dropout, and tells batch_norm layers to use saved statistics instead of batch data.
        # (This affects both the computed scores and the computed gradients, so it is vital)
        model.eval()

        # disable backprop for all the following operations: hopefully this should make them faster
        with torch.no_grad():

            head_to_explain, relation_to_explain, tail_to_explain = sample
            entity_to_explain, target_entity = (relation_to_explain, tail_to_explain) if perspective == "head" else (tail_to_explain, head_to_explain)

            overall_candidate_entities = []

            if perspective == "head":

                step_1_candidate_entities = []
                step_1_samples = []
                for cur_entity in range(0, dataset.num_entities):

                    # do not include the entity to explain, of course
                    if cur_entity == entity_to_explain:
                        continue

                    # if the entity only appears in validation/testing (so its training degree is 0) skip it
                    if dataset.entity_2_degree[cur_entity] < 1:
                        continue

                    # if the training degree exceeds the cap, skip the entity
                    if degree_cap != -1 and dataset.entity_2_degree[cur_entity] > degree_cap:
                        continue

                    # if any facts <cur_entity, relation, *> are in the dataset:
                    if (cur_entity, relation_to_explain) in dataset.to_filter:

                        ## if the relation is *_TO_ONE, ignore any entities for which in train/valid/test,
                        if dataset.relation_2_type[relation_to_explain] in [ONE_TO_ONE, MANY_TO_ONE]:
                            continue

                        ## if <cur_entity, relation, tail> is in the dataset, ignore this entity
                        if tail_to_explain in dataset.to_filter[(cur_entity, relation_to_explain)]:
                            continue

                    step_1_candidate_entities.append(cur_entity)
                    step_1_samples.append((cur_entity, relation_to_explain, tail_to_explain))

                if len(step_1_candidate_entities) == 0:
                    return []

                batch_size = 500
                # if isinstance(model, TransE) and len(step_1_samples) > batch_size:
                batch_scores_array = []
                batch_start = 0
                while batch_start < len(step_1_samples):
                    cur_batch = step_1_samples[batch_start: min(len(step_1_samples), batch_start + batch_size)]
                    cur_batch_all_scores = model.all_scores(samples=numpy.array(cur_batch)).detach().cpu().numpy()
                    batch_scores_array.append(cur_batch_all_scores)
                    batch_start += batch_size
                samples_all_scores = numpy.vstack(batch_scores_array)

                #else:
                #    samples_all_scores = model.all_scores(samples=numpy.array(step_1_samples)).detach().cpu().numpy()

                for i in range(len(step_1_candidate_entities)):
                    cur_candidate_entity = step_1_candidate_entities[i]
                    cur_head, cur_rel, cur_tail = step_1_samples[i]
                    cur_sample_all_scores = samples_all_scores[i]

                    filter_out = dataset.to_filter[(cur_head, cur_rel)] if (cur_head, cur_rel) in dataset.to_filter else []

                    if model.is_minimizer():
                        cur_sample_all_scores[torch.LongTensor(filter_out)] = 1e6
                        cur_sample_target_score_filtered = cur_sample_all_scores[cur_tail]
                        if 1e6 > cur_sample_target_score_filtered > numpy.min(cur_sample_all_scores):
                            overall_candidate_entities.append(cur_candidate_entity)
                    else:
                        cur_sample_all_scores[torch.LongTensor(filter_out)] = -1e6
                        cur_sample_target_score_filtered = cur_sample_all_scores[cur_tail]
                        if -1e6 < cur_sample_target_score_filtered < numpy.max(cur_sample_all_scores):
                            overall_candidate_entities.append(cur_candidate_entity)

            else:
                # todo: this is currently not allowed because we would need to collect (head, relation, entity) for all other entities
                # todo: add an optional boolean "head_prediction" (default=False); if it is true, compute scores for all heads rather than tails
                raise NotImplementedError

        return random.sample(overall_candidate_entities, k=min(k, len(overall_candidate_entities)))
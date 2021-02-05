from typing import Tuple, Any

import numpy
import torch

from dataset import Dataset
from engines.engine import ExplanationEngine
from link_prediction.models.tucker import TuckER
from model import Model

class GradientEngine(ExplanationEngine):

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 epsilon:float):

        ExplanationEngine.__init__(self, model=model, dataset=dataset, hyperparameters=hyperparameters)
        self.epsilon = epsilon

    def simple_removal_explanations(self,
                                    sample_to_explain: Tuple[Any, Any, Any],
                                    perspective: str,
                                    top_k: int):
        """
            Given a sample to explain, and the perspective from which to explain it,
            find the k training samples containing the perspective entity that, if removed (one by one)
            would affect the most the prediction of the sample to explain.

            :param sample_to_explain: the sample to explain in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param top_k: the number of top relevant training samples to return

            :return: an array of k pairs, where each pair is a relevant training sample with its relevance value,
                     sorted by descending relevance
        """

        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # extract all training samples containing the entity to explain
        samples_containing_entity_to_explain = [(h, r, t) for (h, r, t) in self.dataset.train_samples if entity_to_explain_id in [h, t]]
        if len(samples_containing_entity_to_explain) == 0:
            return None

        # this is EXTREMELY important in models with dropout and/or batch normalization.
        # It disables dropout, and tells batch_norm layers to use saved statistics instead of batch data.
        # This affects both the computed scores and the computed gradients.
        self.model.eval()

        target_entity_score = self.model.score(numpy.array([sample_to_explain]))[0]

        # for each training sample containing the entity to explain, compute the relevance by shift
        # (i.e. given the gradient of the training sample with respect to the embedding of the entity to explain,
        # shift the embedding of the entity to explain in the direction of that gradient
        # and measure the improvement of the score of the fact to explain)
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in samples_containing_entity_to_explain])
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]
        for i, current_sample in enumerate(samples_containing_entity_to_explain):

            # the gradient points towards the direction that INCREASES the score
            current_gradient = self.compute_gradient_for(sample=current_sample, entity_to_explain=entity_to_explain_id)

            if perspective=="head":
                # if lesser scores are better, move the embedding in the direction that INCREASES the score of the fact to remove
                if self.model.is_minimizer():
                    head_embeddings_tensor[i] = (head_embeddings_tensor[i] + self.epsilon*current_gradient).cuda()
                # otherwise, if greater scores are better, move the embedding in the direction that DECREASES the score of the fact to remove
                else:
                    head_embeddings_tensor[i] = (head_embeddings_tensor[i] - self.epsilon*current_gradient).cuda()

            # todo: support tail perspective
            else:
                raise Exception("Not supported yet")
                # tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] - self.epsilon*sample_2_gradient[current_sample]).cuda()

        # for each removed fact, compute the score of the fact to explain using the entity embedding accordingly shifted.
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()
        removed_sample_2_perturbed_score = {}
        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i]
            current_sample = samples_containing_entity_to_explain[i]
            removed_sample_2_perturbed_score[current_sample] = current_perturbed_score

        # for each removed fact, compute the relevance as "how much the score of the fact to explain has worsened by removing that fact"
        removed_sample_2_relevance = {}
        for cur_removed_sample in samples_containing_entity_to_explain:
            if self.model.is_minimizer():
                removed_sample_2_relevance[cur_removed_sample] = removed_sample_2_perturbed_score[cur_removed_sample] - target_entity_score
            else:
                removed_sample_2_relevance[cur_removed_sample] = target_entity_score - removed_sample_2_perturbed_score[cur_removed_sample]

        # sort all items by descending relevance
        sorted_samples_with_relevance = sorted(removed_sample_2_relevance.items(), key=lambda x: x[1], reverse=True)
        return sorted_samples_with_relevance if top_k == -1 or top_k < len(sorted_samples_with_relevance) else sorted_samples_with_relevance[:top_k]


    def simple_addition_explanations(self,
                                     sample_to_convert: Tuple[Any, Any, Any],
                                     perspective: str,
                                     samples_to_add: list):
        """
            Given a "sample to convert" (that is, a sample that the model currently does not predict as true,
            and that we want to be predicted as true); given the perspective from which to intervene on it;
            and given and a list of other training samples containing the perspective entity;
            for each sample in the list, compute an esteem of the relevance it would have if added to the perspective entity
            to improve the most the prediction of the sample to convert.

            :param sample_to_convert: the sample that we would like the model to predict as "true",
                                      in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param samples_to_add: the list of samples containing the perspective entity
                                   that we want to analyze the effect of, if added to the perspective entity

            :return: an array of pairs, where each pair is a samples_to_add with the extracted value of relevance,
                     sorted by descending relevance
        """

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_convert
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # this is EXTREMELY important in models with dropout and/or batch normalization.
        # It disables dropout, and tells batch_norm layers to use saved statistics instead of batch data.
        # This affects both the computed scores and the computed gradients.
        self.model.eval()

        target_entity_score, best_entity_score = self._target_and_best_scores_for_sample(numpy.array(sample_to_convert))

        # for each sample to add, compute the score the when the entity to explain gets shifted by an epsilon
        # in the direction that makes the score of the sample to add better
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in range(len(samples_to_add))])
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]
        for i in range(len(samples_to_add)):
            current_sample = samples_to_add[i]

            # the gradient points towards the direction that INCREASES the score
            current_gradient = self.compute_gradient_for(sample=current_sample, entity_to_explain=entity_to_explain_id)

            if perspective=="head":
                if self.model.is_minimizer():
                    head_embeddings_tensor[i] = (head_embeddings_tensor[i] - self.epsilon*current_gradient).cuda()
                else:
                    head_embeddings_tensor[i] = (head_embeddings_tensor[i] + self.epsilon*current_gradient).cuda()

            # todo: support tail perspective
            else:
                raise Exception("Not supported yet")
                #tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] + eps*current_gradient[current_sample]).cuda()

        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()
        added_sample_2_perturbed_score = {}
        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i]
            current_added_sample = samples_to_add[i]
            added_sample_2_perturbed_score[current_added_sample] = current_perturbed_score

        # map each added sample to its relevance - that is, how much the score has improved by "adding" that sample
        sample_2_relevance = {}
        for cur_added_sample in samples_to_add:
            if self.model.is_minimizer():
                sample_2_relevance[cur_added_sample] = target_entity_score - added_sample_2_perturbed_score[cur_added_sample]
            else:
                sample_2_relevance[cur_added_sample] = added_sample_2_perturbed_score[cur_added_sample] - target_entity_score

        sorted_samples_with_relevance = sorted(sample_2_relevance.items(), key=lambda x: x[1], reverse=True)


        return sorted_samples_with_relevance, \
               added_sample_2_perturbed_score, \
               target_entity_score, \
               best_entity_score


    def nple_addition_explanations(self,
                                   sample_to_convert: Tuple[Any, Any, Any],
                                   perspective: str,
                                   samples_to_add: list,
                                   n: int):
        """
            Given a "sample to convert" (that is, a sample that the model currently does not predict as true,
            and that we want to be predicted as true); given the perspective from which to intervene on it;
            given and a list of other training samples containing the perspective entity;
            and given the length "n" of combinations of training samples to try;
            for each combination of length "n" of the samples in the list,
            estimate the relevance that adding that combination of facts to the perspective entity
            would have to improve the prediction of the sample to convert.

            :param sample_to_convert: the sample that we would like the model to predict as "true",
                                      in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param samples_to_add: the list of samples containing the perspective entity
                                   that we want to analyze the effect of, if added to the perspective entity
            :param n: length of combinations of samples to add to the perspective entity

            :return: an array of pairs, where each pair is a n n-long combination of samples_to_add
                     coupled with with the corresponding extracted value of relevance, sorted by descending relevance
        """


        # if there are less than n items in samples_to_add
        # it is not possible to try combinations of length n of samples_to_add
        if len(samples_to_add) < n:
            return []

        # this is EXTREMELY important in models with dropout and/or batch normalization.
        # It disables dropout, and tells batch_norm layers to use saved statistics instead of batch data.
        # This affects both the computed scores and the computed gradients.
        self.model.eval()

        target_entity_score, best_entity_score = self._target_and_best_scores_for_sample(numpy.array(sample_to_convert))

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_convert
        entity_to_explain_id = head_id if perspective == "head" else tail_id


        # for each sample to add, compute the gradient of its own score with respect to the embedding of the entity to explain
        # and compute the mapping sample to add -> gradient
        sample_to_add_2_gradient = {}
        for cur_sample_to_add in samples_to_add:
            sample_to_add_2_gradient[cur_sample_to_add] = self.compute_gradient_for(sample=cur_sample_to_add,
                                                                                    entity_to_explain=entity_to_explain_id)

        # extract all n-long different combinations of samples
        samples_nples = self._extract_sample_nples(samples=samples_to_add, n=n)
        nple_2_gradient = {}

        for i in range(len(samples_nples)):
            cur_samples_nple = samples_nples[i]
            cur_gradient = 0
            for cur_sample_to_add in cur_samples_nple:
                cur_gradient += sample_to_add_2_gradient[cur_sample_to_add]
            nple_2_gradient[tuple(cur_samples_nple)] = cur_gradient

        # for each couple training sample containing the entity to explain,
        # shift the entity to explain in the gradient direction and verify the score improvement
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in range(len(samples_nples))])
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]

        for i in range(len(samples_nples)):
            current_nple = samples_nples[i]
            if perspective=="head":
                if self.model.is_minimizer():
                    head_embeddings_tensor[i] = (head_embeddings_tensor[i] - self.epsilon*nple_2_gradient[current_nple]).cuda()
                else:
                    head_embeddings_tensor[i] = (head_embeddings_tensor[i] + self.epsilon*nple_2_gradient[current_nple]).cuda()
            # todo: support tail perspective
            else:
                raise Exception("Not supported yet!")
                #tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] + self.epsilon*nple_2_gradient[current_nple]).cuda()
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor)

        added_nple_2_perturbed_score = {}
        added_nple_2_relevance = {}

        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i].detach().cpu().numpy()
            cur_added_nple = samples_nples[i]
            added_nple_2_perturbed_score[tuple(cur_added_nple)] = current_perturbed_score

            if self.model.is_minimizer():
                added_nple_2_relevance[tuple(cur_added_nple)] = target_entity_score - current_perturbed_score
            else:
                added_nple_2_relevance[tuple(cur_added_nple)] = current_perturbed_score - target_entity_score

        sorted_nples_with_relevance = sorted(added_nple_2_relevance.items(), key=lambda x:x[1], reverse=True)

        return sorted_nples_with_relevance, \
               added_nple_2_perturbed_score, \
               target_entity_score, \
               best_entity_score

    def compute_gradient_for(self,
                             sample: Tuple[Any, Any, Any],
                             entity_to_explain: int):

        entity_dimension = self.model.entity_dimension if isinstance(self.model, TuckER) else self.model.dimension
        relation_dimension = self.model.relation_dimension if isinstance(self.model, TuckER) else self.model.dimension

        sample_head, sample_relation, sample_tail = sample
        sample_head_embedding = self.model.entity_embeddings[sample_head].detach().reshape(1, entity_dimension)
        sample_relation_embedding = self.model.relation_embeddings[sample_relation].detach().reshape(1, relation_dimension)
        sample_tail_embedding = self.model.entity_embeddings[sample_tail].detach().reshape(1, entity_dimension)

        cur_entity_to_explain_embedding = sample_head_embedding if entity_to_explain == sample_head else sample_tail_embedding
        cur_entity_to_explain_embedding.requires_grad = True

        current_score = self.model.score_embeddings(sample_head_embedding, sample_relation_embedding, sample_tail_embedding)
        current_score.backward()

        current_gradient = cur_entity_to_explain_embedding.grad[0]
        cur_entity_to_explain_embedding.grad = None  # reset the gradient, just to be sure

        return current_gradient

    def compute_gradient_for_embeddings(self, head_embedding, rel_embedding, tail_embedding, perspective):
        entity_dimension = self.model.entity_dimension if isinstance(self.model, TuckER) else self.model.dimension
        relation_dimension = self.model.relation_dimension if isinstance(self.model, TuckER) else self.model.dimension

        head_embedding = head_embedding.detach().reshape(entity_dimension)
        rel_embedding = rel_embedding.detach().reshape(1, relation_dimension)
        tail_embedding = tail_embedding.detach().reshape(1, entity_dimension)

        current_gradient = None
        if perspective == "head":
            head_embedding.requires_grad = True
            current_score = self.model.score_embeddings(head_embedding, rel_embedding, tail_embedding)
            current_score.backward()
            current_gradient = head_embedding.grad[0].detach()
            head_embedding.grad = None
        else:
            tail_embedding.requires_grad = True
            current_score = self.model.score_embeddings(head_embedding, rel_embedding, tail_embedding)
            current_score.backward()
            current_gradient = tail_embedding.grad[0].detach()
            tail_embedding.grad = None

        return current_gradient


    def projection_size(self, a: numpy.array, b: numpy.array):
        """
            Compute the size of the projection of b on the direction of a
            :param a:
            :param b:
            :return:
        """

        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.cpu().numpy()

        return numpy.dot(a, b) / numpy.linalg.norm(a)


    def _target_and_best_scores_for_sample(self, sample):
        head_id, relation_id, tail_id = sample
        all_scores = self.model.all_scores(numpy.array([sample])).detach().cpu().numpy()[0]
        target_score = all_scores[tail_id]
        filter_out = self.dataset.to_filter[(head_id, relation_id)] if (head_id, relation_id) in self.dataset.to_filter else []

        if self.model.is_minimizer():
            all_scores[filter_out] = 1e6
            best_score = numpy.min(all_scores)
        else:
            all_scores[filter_out] = -1e6
            best_score = numpy.max(all_scores)

        return target_score, best_score
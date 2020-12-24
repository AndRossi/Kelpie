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

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # extract all training samples containing the entity to explain
        samples_containing_entity_to_explain = [(h, r, t) for (h, r, t) in self.dataset.train_samples if entity_to_explain_id in [h, t]]
        if len(samples_containing_entity_to_explain) == 0:
            return None

        # for each training sample containing the entity to explain, compute the relevance by projection
        sample_to_remove_2_gradients = {}
        for cur_sample_to_remove in samples_containing_entity_to_explain:
            sample_to_remove_2_gradients[cur_sample_to_remove] = self.compute_gradients_for(cur_sample_to_remove)

        # for each training sample containing the entity to explain, compute the relevance by shift
        # (i.e. given the gradient of the training sample with respect to the embedding of the entity to explain,
        # shift the embedding of the entity to explain in the direction of that gradient
        # and measure the improvement of the score of the fact to explain)
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in samples_containing_entity_to_explain])
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]

        for i in range(len(samples_containing_entity_to_explain)):
            cur_sample_to_remove = samples_containing_entity_to_explain[i]
            head_grad, rel_grad, tail_grad = sample_to_remove_2_gradients[cur_sample_to_remove]

            head_embeddings_tensor[i] = (head_embeddings_tensor[i] - self.epsilon * head_grad).cuda()
            rel_embeddings_tensor[i] = (rel_embeddings_tensor[i] - self.epsilon * rel_grad).cuda()
            tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] - self.epsilon * tail_grad).cuda()

        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()

        sample_2_relevance_by_score = {}
        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i]
            current_sample = samples_containing_entity_to_explain[i]
            sample_2_relevance_by_score[current_sample] = current_perturbed_score

        # TODO: THIS ONLY WORKS FOR MAXIMIZING MODELS SO FAR (SHOULD BE reverse=True FOR MINIMIZING MODELS)
        most_relevant_samples = sorted(sample_2_relevance_by_score.items(), key=lambda x:x[1], reverse=False)

        if top_k == -1 or top_k < len(most_relevant_samples):
            return most_relevant_samples
        else:
            return most_relevant_samples[:top_k]


    def simple_addition_explanations(self,
                    sample_to_explain: Tuple[Any, Any, Any],
                    perspective: str,
                    samples_to_add: list,
                                     eps=None):

        if eps is None:
            eps = self.epsilon

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        target_entity_score, best_entity_score = self._target_and_best_scores_for_sample(numpy.array(sample_to_explain))

        # compute the gradient of the fact to explain with respect to the embedding of the entity to explain
        # gradient = self.compute_gradient_for(sample=sample_to_explain, entity_to_explain=entity_to_explain_id)

        sample_2_gradients = {}
        #sample_2_relevance_by_projection = {}
        for current_sample in samples_to_add:
            cur_sample_grads = self.compute_gradients_for(sample=current_sample)
            sample_2_gradients[tuple(current_sample)] = cur_sample_grads
            # for each sample to add, compute the relevance by projection
            # projection_size = self.projection_size(gradient, current_gradient)
            # sample_2_relevance_by_projection[tuple(current_sample)] = projection_size

        #most_relevant_samples_by_projection = sorted(sample_2_relevance_by_projection.items(), key=lambda x:x[1], reverse=True)

        # for each sample to add featuring the entity to explain,
        # compute and store how the score of the sample to explain changes
        # when as entity to explain gets shifted by an epsilon towards the gradient of the score to the sample to add
        added_sample_2_perturbed_score = {}
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in range(len(samples_to_add))])

        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]
        for i in range(len(samples_to_add)):
            current_sample = samples_to_add[i]
            cur_head_grad, cur_rel_grad, cur_tail_grad = sample_2_gradients[current_sample]
            # todo: + or - depending on whether the model maximizes or minimizes the score
            head_embeddings_tensor[i] = (head_embeddings_tensor[i] + eps*cur_head_grad).cuda()
            rel_embeddings_tensor[i] = (rel_embeddings_tensor[i] + eps*cur_rel_grad).cuda()
            tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] + eps*cur_tail_grad).cuda()

        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor)
        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i].detach().cpu().numpy()
            cur_added_sample = samples_to_add[i]
            added_sample_2_perturbed_score[cur_added_sample] = current_perturbed_score

        # compute the relevance of each sample to add,
        # based on the obtained score perturbation on the sample to explain.
        # Relevance = score that the target tail would obtain with perturbation - score that it would have obtained originally
        sample_2_relevance = {}
        for cur_added_sample in samples_to_add:
            # todo: depends on whether the model maximizes or minimizes the score
            sample_2_relevance[cur_added_sample] = added_sample_2_perturbed_score[cur_added_sample] - target_entity_score

        # todo: "reverse" depends on whether the model maximizes or minimizes the score
        sorted_samples_with_relevance = sorted(sample_2_relevance.items(), key=lambda x:x[1], reverse=True)

        return sorted_samples_with_relevance, \
               added_sample_2_perturbed_score, \
               target_entity_score, \
               best_entity_score


    def nple_addition_explanations(self,
                                   sample_to_explain: Tuple[Any, Any, Any],
                                   perspective: str,
                                   samples_to_add: list,
                                   n: int):

        # if there are less than n items in samples_to_add
        # it is not possible to try combinations of length n of samples_to_add
        if len(samples_to_add) < n:
            return []

        target_entity_score, best_entity_score = self._target_and_best_scores_for_sample(numpy.array(sample_to_explain))

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # compute the gradient of the score of fact to explain with respect to the embedding of the entity to explain
        # gradient = self.compute_gradient_for(sample=sample_to_explain, entity_to_explain=entity_to_explain_id)

        # for each sample to add, compute the gradient of its own score with respect to the embedding of the entity to explain
        # and compute the mapping sample to add -> gradient
        sample_to_add_2_gradient = {}
        for cur_sample_to_add in samples_to_add:
            sample_to_add_2_gradient[cur_sample_to_add] = self.compute_gradient_for(sample=cur_sample_to_add, entity_to_explain=entity_to_explain_id)

        # extract all n-long different combinations of samples
        samples_nples = self._extract_sample_nples(samples=samples_to_add, n=n)
        nple_2_gradient = {}

        #nple_2_relevance_by_projection = {}
        for i in range(len(samples_nples)):
            cur_samples_nple = samples_nples[i]
            cur_gradient = 0
            for cur_sample_to_add in cur_samples_nple:
                cur_gradient += sample_to_add_2_gradient[cur_sample_to_add]
            nple_2_gradient[tuple(cur_samples_nple)] = cur_gradient
            # projection_size = self.projection_size(gradient, cur_gradient)
            # nple_2_relevance_by_projection[tuple(cur_samples_nple)] = projection_size

        # most_relevant_nples_by_projection = sorted(nple_2_relevance_by_projection.items(), key=lambda x:x[1], reverse=True)

        # for each couple training sample containing the entity to explain,
        # shift the entity to explain in the gradient direction and verify the score improvement
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in range(len(samples_nples))])

        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]

        for i in range(len(samples_nples)):
            current_nple = samples_nples[i]
            if perspective=="head":
                # todo: depends on whether the model maximizes or minimizes the score
                head_embeddings_tensor[i] = (head_embeddings_tensor[i] + self.epsilon*nple_2_gradient[current_nple]).cuda()
            else:
                # todo: depends on whether the model maximizes or minimizes the score
                tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] + self.epsilon*nple_2_gradient[current_nple]).cuda()
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor)

        added_nple_2_perturbed_score = {}
        added_nple_2_relevance = {}

        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i].detach().cpu().numpy()
            cur_added_nple = samples_nples[i]
            added_nple_2_perturbed_score[tuple(cur_added_nple)] = current_perturbed_score
            # todo: depends on whether the model maximizes or minimizes the score
            added_nple_2_relevance[tuple(cur_added_nple)] = current_perturbed_score - target_entity_score

        sorted_nples_with_relevance = sorted(added_nple_2_relevance.items(), key=lambda x:x[1], reverse=True)

        return sorted_nples_with_relevance, \
               added_nple_2_perturbed_score, \
               target_entity_score, \
               best_entity_score

    def compute_gradients_for(self,
                             sample: Tuple[Any, Any, Any]):

        entity_dimension = self.model.entity_dimension if isinstance(self.model, TuckER) else self.model.dimension
        relation_dimension = self.model.relation_dimension if isinstance(self.model, TuckER) else self.model.dimension

        sample_head, sample_relation, sample_tail = sample
        sample_head_embedding = self.model.entity_embeddings[sample_head].detach().reshape(1, entity_dimension)
        sample_relation_embedding = self.model.relation_embeddings[sample_relation].detach().reshape(1, relation_dimension)
        sample_tail_embedding = self.model.entity_embeddings[sample_tail].detach().reshape(1, entity_dimension)

        sample_head_embedding.requires_grad = True
        sample_relation_embedding.requires_grad = True
        sample_tail_embedding.requires_grad = True

        current_score = self.model.score_embeddings(sample_head_embedding, sample_relation_embedding, sample_tail_embedding)
        current_score.backward()

        sample_head_embedding_grad = sample_head_embedding.grad[0]
        sample_relation_embedding_grad = sample_relation_embedding.grad[0]
        sample_tail_embedding_grad = sample_tail_embedding.grad[0]

        # reset the gradient, just to be sure
        sample_head_embedding.grad, sample_relation_embedding.grad, sample_tail_embedding.grad = None, None, None

        return sample_head_embedding_grad, sample_relation_embedding_grad, sample_tail_embedding_grad

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

    # todo: this should depend on whether the model maximises or minimizes scores
    def _compute_relevance(self, perturbed_score, original_score):
        return perturbed_score - original_score


    def _target_and_best_scores_for_sample(self, sample):
        head_id, relation_id, tail_id = sample
        all_scores = self.model.all_scores(numpy.array([sample])).detach().cpu().numpy()[0]
        target_score = all_scores[tail_id]

        filter_out = self.dataset.to_filter[(head_id, relation_id)]
        all_scores[filter_out] = -1e6         # todo: make it huge if the model wants to minimize scores
        best_score = numpy.max(all_scores)      # todo: min if the model wants to minimize scores
        return target_score, best_score


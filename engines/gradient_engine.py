from typing import Tuple, Any

import numpy
import torch

from dataset import Dataset
from engines.engine import ExplanationEngine
from model import Model

class GradientEngine(ExplanationEngine):

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 epsilon:float):

        ExplanationEngine.__init__(self, model=model, dataset=dataset, hyperparameters=hyperparameters)
        self.epsilon = epsilon

    def simple_removal_explanations(self, sample_to_explain: Tuple[Any, Any, Any], perspective: str):

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # compute the gradient of the fact to explain with respect to the embedding of the entity to explain
        gradient = self.compute_gradient_for(sample=sample_to_explain, entity_to_explain=entity_to_explain_id)

        # extract all training samples containing the entity to explain
        samples_containing_entity_to_explain = [(h, r, t) for (h, r, t) in self.dataset.train_samples if entity_to_explain_id in [h, t]]
        if len(samples_containing_entity_to_explain) == 0:
            return None

        # for each training sample containing the entity to explain, compute the relevance by projection
        sample_2_relevance_by_projection = {}
        sample_2_gradient = {}
        for current_sample in samples_containing_entity_to_explain:
            current_gradient = self.compute_gradient_for(current_sample, entity_to_explain_id)
            projection_size = self.projection_size(gradient, current_gradient)

            sample_2_gradient[current_sample] = current_gradient
            sample_2_relevance_by_projection[current_sample] = projection_size
        most_relevant_samples_by_projection = sorted(sample_2_relevance_by_projection.items(), key=lambda x:x[1], reverse=True)

        # for each training sample containing the entity to explain, compute the relevance by shift
        # (i.e. given the gradient of the training sample with respect to the embedding of the entity to explain,
        # shift the embedding of the entity to explain in the direction of that gradient
        # and measure the improvement of the score of the fact to explain)
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in most_relevant_samples_by_projection])
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]
        for i in range(len(most_relevant_samples_by_projection)):
            item = most_relevant_samples_by_projection[i]
            current_sample = item[0]
            if perspective=="head":
                head_embeddings_tensor[i] = (head_embeddings_tensor[i] + self.epsilon*sample_2_gradient[current_sample]).cuda()
            else:
                tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] + self.epsilon*sample_2_gradient[current_sample]).cuda()
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()

        sample_2_relevance_by_score = {}
        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i]
            current_sample = most_relevant_samples_by_projection[i][0]
            sample_2_relevance_by_score[current_sample] = current_perturbed_score

        most_relevant_samples = sorted(sample_2_relevance_by_score.items(), key=lambda x:x[1], reverse=True)
        return most_relevant_samples


    def simple_addition_explanations(self,
                    sample_to_explain: Tuple[Any, Any, Any],
                    perspective: str,
                    samples_to_add: list):

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # compute the gradient of the fact to explain with respect to the embedding of the entity to explain
        gradient = self.compute_gradient_for(sample=sample_to_explain, entity_to_explain=entity_to_explain_id)

        # for each sample to add, compute the relevance by projection
        sample_2_relevance_by_projection = {}
        sample_2_gradient = {}
        for current_sample in samples_to_add:
            current_gradient = self.compute_gradient_for(sample=current_sample, entity_to_explain=entity_to_explain_id)
            projection_size = self.projection_size(gradient, current_gradient)

            sample_2_gradient[tuple(current_sample)] = current_gradient
            sample_2_relevance_by_projection[tuple(current_sample)] = projection_size

        most_relevant_samples_by_projection = sorted(sample_2_relevance_by_projection.items(), key=lambda x:x[1], reverse=True)

        # for each sample to add containing the entity to explain, compute the relevance by shift
        # (i.e. given the gradient of the training sample with respect to the embedding of the entity to explain,
        # shift the embedding of the entity to explain in the direction of that gradient
        # and measure the improvement of the score of the fact to explain)
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in range(len(most_relevant_samples_by_projection))])
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]
        for i in range(len(most_relevant_samples_by_projection)):
            current_sample = most_relevant_samples_by_projection[i][0]
            if perspective=="head":
                head_embeddings_tensor[i] = (head_embeddings_tensor[i] + self.epsilon*sample_2_gradient[current_sample]).cuda()
            else:
                tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] + self.epsilon*sample_2_gradient[current_sample]).cuda()
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor)

        sample_2_relevance_by_score = {}
        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i].detach().cpu().numpy()
            current_sample = most_relevant_samples_by_projection[i][0]
            sample_2_relevance_by_score[current_sample] = current_perturbed_score

        most_relevant_samples = sorted(sample_2_relevance_by_score.items(), key=lambda x:x[1], reverse=True)
        return most_relevant_samples

    def couple_addition_explanations(self,
                                     sample_to_explain: Tuple[Any, Any, Any],
                                     perspective: str,
                                     samples_to_add: list):

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # compute the gradient of the fact to explain with respect to the embedding of the entity to explain
        gradient = self.compute_gradient_for(sample=sample_to_explain, entity_to_explain=entity_to_explain_id)

        # for each couple of samples to add, compute the relevance by projection
        couple_2_relevance_by_projection = {}
        couple_2_gradient = {}

        samples_couples = self._extract_sample_couples(samples=samples_to_add)
        for i in range(len(samples_couples)):
            cur_couple = samples_couples[i]
            cur_gradient_1 = self.compute_gradient_for(sample=cur_couple[0], entity_to_explain=entity_to_explain_id)
            cur_gradient_2 = self.compute_gradient_for(sample=cur_couple[1], entity_to_explain=entity_to_explain_id)
            cur_gradient = cur_gradient_1+cur_gradient_2

            projection_size = self.projection_size(gradient, cur_gradient)

            couple_2_gradient[tuple(cur_couple)] = cur_gradient
            couple_2_relevance_by_projection[tuple(cur_couple)] = projection_size

        most_relevant_couples_by_projection = sorted(couple_2_relevance_by_projection.items(), key=lambda x:x[1], reverse=True)

        # for each couple training sample containing the entity to explain,
        # shift the entity to explain in the gradient direction and verify the score improvement
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in range(len(couple_2_relevance_by_projection))])
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]
        for i in range(len(most_relevant_couples_by_projection)):
            item = most_relevant_couples_by_projection[i]
            current_couple = item[0]
            if perspective=="head":
                head_embeddings_tensor[i] = (head_embeddings_tensor[i] + self.epsilon*couple_2_gradient[current_couple]).cuda()
            else:
                tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] + self.epsilon*couple_2_gradient[current_couple]).cuda()
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor)

        couple_2_relevance_by_score = {}
        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i].detach().cpu().numpy()
            current_couple = most_relevant_couples_by_projection[i][0]
            couple_2_relevance_by_score[tuple(current_couple)] = current_perturbed_score

        most_relevant_couples = sorted(couple_2_relevance_by_score.items(), key=lambda x:x[1], reverse=True)
        return most_relevant_couples


    def compute_gradient_for(self,
                             sample: Tuple[Any, Any, Any],
                             entity_to_explain: int):

        sample_head, sample_relation, sample_tail = sample
        sample_head_embedding = self.model.entity_embeddings[sample_head].detach().reshape(1, self.model.dimension)
        sample_relation_embedding = self.model.relation_embeddings[sample_relation].detach().reshape(1, self.model.dimension)
        sample_tail_embedding = self.model.entity_embeddings[sample_tail].detach().reshape(1, self.model.dimension)

        cur_entity_to_explain_embedding = sample_head_embedding if entity_to_explain == sample_head else sample_tail_embedding
        cur_entity_to_explain_embedding.requires_grad = True

        current_score = self.model.score_embeddings(sample_head_embedding, sample_relation_embedding, sample_tail_embedding)
        current_score.backward()
        current_gradient = cur_entity_to_explain_embedding.grad[0].detach()
        cur_entity_to_explain_embedding.grad = None  # reset the gradient, just to be sure

        return current_gradient

    def compute_gradient_for_embeddings(self, head_embedding, rel_embedding, tail_embedding, perspective):

        head_embedding = head_embedding.detach().reshape(1, self.model.dimension)
        rel_embedding = rel_embedding.detach().reshape(1, self.model.dimension)
        tail_embedding = tail_embedding.detach().reshape(1, self.model.dimension)

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
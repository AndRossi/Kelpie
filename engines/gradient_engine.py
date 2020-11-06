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

        output_details_path = "output_details.txt"
        output_details_lines = []
        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        fact = self.dataset.sample_to_fact(sample_to_explain)
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

        #for cur_sample, cur_relevance in most_relevant_samples:
        #    cur_fact = self.dataset.sample_to_fact(cur_sample)
        #    cur_projection = sample_2_relevance_by_projection[cur_sample]
        #    output_details_lines.append(";".join(fact)+ "," + ";".join(cur_fact) + "," + str(cur_projection) + "," + str(cur_relevance) + "\n")
        #with open(output_details_path, "a") as outfile:
        #    outfile.writelines(output_details_lines)

        return most_relevant_samples

    def nple_addition_explanations(self,
                                   sample_to_explain: Tuple[Any, Any, Any],
                                   perspective: str,
                                   samples_to_add: list,
                                   n: int):

        # if there are less than n items in samples_to_add
        # it is not possible to try combinations of length n of samples_to_add
        if len(samples_to_add) < n:
            return []

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # compute the gradient of the score of fact to explain with respect to the embedding of the entity to explain
        gradient = self.compute_gradient_for(sample=sample_to_explain, entity_to_explain=entity_to_explain_id)

        # for each sample to add, compute the gradient of its own score with respect to the embedding of the entity to explain
        # and compute the mapping sample to add -> gradient
        sample_to_add_2_gradient = {}
        for cur_sample_to_add in samples_to_add:
            sample_to_add_2_gradient[cur_sample_to_add] = self.compute_gradient_for(sample=cur_sample_to_add, entity_to_explain=entity_to_explain_id)

        # extract all n-long different combinations of samples
        samples_nples = self._extract_sample_nples(samples=samples_to_add, n=n)
        nple_2_gradient = {}
        nple_2_relevance_by_projection = {}
        for i in range(len(samples_nples)):
            cur_samples_nple = samples_nples[i]

            cur_gradient = 0
            for cur_sample_to_add in cur_samples_nple:
                cur_gradient += sample_to_add_2_gradient[cur_sample_to_add]

            projection_size = self.projection_size(gradient, cur_gradient)

            nple_2_gradient[tuple(cur_samples_nple)] = cur_gradient
            nple_2_relevance_by_projection[tuple(cur_samples_nple)] = projection_size

        most_relevant_nples_by_projection = sorted(nple_2_relevance_by_projection.items(), key=lambda x:x[1], reverse=True)

        # for each couple training sample containing the entity to explain,
        # shift the entity to explain in the gradient direction and verify the score improvement
        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in range(len(nple_2_relevance_by_projection))])

        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]
        for i in range(len(most_relevant_nples_by_projection)):
            item = most_relevant_nples_by_projection[i]
            current_nple = item[0]
            if perspective=="head":
                head_embeddings_tensor[i] = (head_embeddings_tensor[i] + self.epsilon*nple_2_gradient[current_nple]).cuda()
            else:
                tail_embeddings_tensor[i] = (tail_embeddings_tensor[i] + self.epsilon*nple_2_gradient[current_nple]).cuda()
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor)

        nple_2_relevance_by_score = {}
        for i in range(len(perturbed_scores)):
            current_perturbed_score = perturbed_scores[i].detach().cpu().numpy()
            current_nple = most_relevant_nples_by_projection[i][0]
            nple_2_relevance_by_score[tuple(current_nple)] = current_perturbed_score

        most_relevant_nples = sorted(nple_2_relevance_by_score.items(), key=lambda x:x[1], reverse=True)
        return most_relevant_nples


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

        current_gradient = cur_entity_to_explain_embedding.grad[0]
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

from typing import Tuple, Any

import numpy
import torch

from dataset import Dataset
from engines.engine import ExplanationEngine
from model import Model

class DataPoisoningEngine(ExplanationEngine):

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 epsilon:float):

        ExplanationEngine.__init__(self, model=model, dataset=dataset, hyperparameters=hyperparameters)
        self.epsilon = epsilon
        self.lambd = 0.1


    def simple_removal_explanations(self,
                                    sample_to_explain: Tuple[Any, Any, Any],
                                    perspective: str):

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # compute the gradient of the fact to explain with respect to the embedding of the entity to explain
        gradient = self.compute_gradient_for(sample=sample_to_explain, entity_to_explain=entity_to_explain_id)

        # extract all training samples containing the entity to explain
        samples_containing_entity_to_explain = [(h, r, t) for (h, r, t) in self.dataset.train_samples if entity_to_explain_id in [h, t]]
        if len(samples_containing_entity_to_explain) == 0:
            return None

        # move the embedding of the entity to explain in direction OPPOSITE to the gradient
        # (that is, WORSEN the score of the fact to explain)
        perturbed_entity_to_explain_embedding = self.model.entity_embeddings[entity_to_explain_id].detach() - self.epsilon * gradient.detach()


        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in samples_containing_entity_to_explain])
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]


        original_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()


        for i in range(len(samples_containing_entity_to_explain)):
            cur_sample_containing_entity_to_explain = samples_containing_entity_to_explain[i]
            if cur_sample_containing_entity_to_explain[0] == entity_to_explain_id:
                head_embeddings_tensor[i] = perturbed_entity_to_explain_embedding
            else:
                assert(cur_sample_containing_entity_to_explain[2] == entity_to_explain_id)
                tail_embeddings_tensor[i] = perturbed_entity_to_explain_embedding

        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()

        # now for each training sample containing the entity to explain you have
        # both the original score and the score computed with the perturbed embedding
        # so you can compute the relevance of that training sample as original_score - lambda * perturbed_score
        # (that is: we have worsened the fact to explain. The more a training fact was worsened as well, the more relevant it was.)
        sample_2_relevance = {}
        for i in range(len(samples_containing_entity_to_explain)):
            sample_2_relevance[tuple(samples_containing_entity_to_explain[i])] = original_scores[i] - self.lambd * perturbed_scores[i]

        most_relevant_samples = sorted(sample_2_relevance.items(), key=lambda x:x[1], reverse=True)
        return most_relevant_samples



    def simple_addition_explanations(self,
                    sample_to_explain: Tuple[Any, Any, Any],
                    perspective: str,
                    samples_to_add: list):
        output_details_path = "output_details.csv"

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id
        fact = self.dataset.sample_to_fact(sample_to_explain)

        # compute the gradient of the fact to explain with respect to the embedding of the entity to explain
        # and move the embedding of the entity to explain in direction OPPOSITE to the gradient
        # (that is, WORSEN the score of the fact to explain)
        gradient = self.compute_gradient_for(sample=sample_to_explain, entity_to_explain=entity_to_explain_id)
        perturbed_entity_to_explain_embedding = self.model.entity_embeddings[entity_to_explain_id].detach() - self.epsilon * gradient.detach()

        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in samples_to_add])
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]

        original_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()

        for i in range(len(samples_to_add)):
            cur_sample_containing_entity_to_explain = samples_to_add[i]
            if cur_sample_containing_entity_to_explain[0] == entity_to_explain_id:
                head_embeddings_tensor[i] = perturbed_entity_to_explain_embedding
            else:
                assert(cur_sample_containing_entity_to_explain[2] == entity_to_explain_id)
                tail_embeddings_tensor[i] = perturbed_entity_to_explain_embedding
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()

        # now for each sample to add, you have both the original score and the score computed with the perturbed embedding.
        # The relevance of that sample, if added to the training set, is
        # "perturbed_score - lambda * original score"
        sample_2_relevance = {}
        for i in range(len(samples_to_add)):
            sample_2_relevance[tuple(samples_to_add[i])] = original_scores[i] - self.lambd * perturbed_scores[i]


        most_relevant_samples = sorted(sample_2_relevance.items(), key=lambda x:x[1], reverse=True)
        output_details_lines = []

        for cur_sample, cur_relevance in most_relevant_samples:
            cur_fact = self.dataset.sample_to_fact(cur_sample)
            output_details_lines.append(";".join(fact)+ "," + ";".join(cur_fact) + "," + str(cur_relevance) + "\n")
        with open(output_details_path, "a") as outfile:
            outfile.writelines(output_details_lines)

        return most_relevant_samples


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

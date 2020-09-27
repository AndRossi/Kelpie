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

    def simple_removal_explanations(self,
                    sample_to_explain: numpy.array,
                    perspective: str):

        head_id, relation_id, tail_id = sample_to_explain

        # get the embedding of the head entity, of the relation, and of the tail entity of the fact to explain
        head_embedding = self.model.entity_embeddings[head_id].detach().reshape(1, self.model.dimension)
        relation_embedding = self.model.relation_embeddings[relation_id].detach().reshape(1, self.model.dimension)
        tail_embedding = self.model.entity_embeddings[tail_id].detach().reshape(1, self.model.dimension)

        # identify the entity to explain and set set the requires_grad flag of its embedding to true
        entity_to_explain_id = head_id if perspective == "head" else tail_id
        entity_to_explain_embedding = head_embedding if perspective == "head" else tail_embedding
        entity_to_explain_embedding.requires_grad=True

        # compute the score of the fact to explain
        # and compute its gradient with respect to the embedding of the entity to explain
        score = self.model.score_embeddings(head_embedding, relation_embedding, tail_embedding)
        score.backward()
        gradient = entity_to_explain_embedding.grad[0].detach().cpu().numpy()
        entity_to_explain_embedding.grad = None         # reset the gradient, just to be sure

        # extract all training samples containing the entity to explain
        samples_containing_entity_to_explain = numpy.array([(h, r, t) for (h, r, t) in self.dataset.train_samples
                                                            if entity_to_explain_id in [h, t]])
        if len(samples_containing_entity_to_explain) == 0:
            return None

        # for each training sample containing the entity to explain, compute the relevance by projection
        sample_2_relevance_by_projection = {}
        sample_2_gradient = {}
        for i in range(samples_containing_entity_to_explain.shape[0]):
            current_sample = samples_containing_entity_to_explain[i]

            cur_head_embedding = self.model.entity_embeddings[current_sample[0]].detach().reshape(1, self.model.dimension)
            cur_relation_embedding = self.model.relation_embeddings[current_sample[1]].detach().reshape(1, self.model.dimension)
            cur_tail_embedding = self.model.entity_embeddings[current_sample[2]].detach().reshape(1, self.model.dimension)
            cur_entity_to_explain_embedding = cur_head_embedding if perspective == "head" else cur_tail_embedding
            cur_entity_to_explain_embedding.requires_grad = True

            current_score = self.model.score_embeddings(cur_head_embedding, cur_relation_embedding, cur_tail_embedding)
            current_score.backward()
            current_gradient = cur_entity_to_explain_embedding.grad[0]
            projection_size = self.projection_size(gradient, current_gradient.cpu().numpy())

            sample_2_gradient[tuple(current_sample)] = current_gradient
            sample_2_relevance_by_projection[tuple(current_sample)] = projection_size

            cur_entity_to_explain_embedding.grad = None     # reset the gradient, just to be sure

        most_relevant_samples_by_projection = sorted(sample_2_relevance_by_projection.items(), key=lambda x:x[1], reverse=True)

        samples = numpy.array([(head_id, relation_id, tail_id) for _ in range(len(most_relevant_samples_by_projection))])


        # for each training sample containing the entity to explain,
        # shift the entity to explain in the gradient direction and verify the score improvement
        head_embeddings_tensor = self.model.entity_embeddings[samples[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples[:, 2]]
        for i in range(len(most_relevant_samples_by_projection)):
            item = most_relevant_samples_by_projection[i]
            current_sample = item[0]
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


    def simple_addition_explanations(self,
                    sample_to_explain: numpy.array,
                    perspective: str,
                    samples_to_add: numpy.array):

        head_id, relation_id, tail_id = sample_to_explain

        # get the embedding of the head entity, of the relation, and of the tail entity of the fact to explain
        head_embedding = self.model.entity_embeddings[head_id].detach().reshape(1, self.model.dimension)
        relation_embedding = self.model.relation_embeddings[relation_id].detach().reshape(1, self.model.dimension)
        tail_embedding = self.model.entity_embeddings[tail_id].detach().reshape(1, self.model.dimension)

        # identify the entity to explain and set set the requires_grad flag of its embedding to true
        entity_to_explain_id = head_id if perspective == "head" else tail_id
        entity_to_explain_embedding = head_embedding if perspective == "head" else tail_embedding
        entity_to_explain_embedding.requires_grad=True

        # compute the score of the fact to explain
        # and compute its gradient with respect to the embedding of the entity to explain
        score = self.model.score_embeddings(head_embedding, relation_embedding, tail_embedding)
        score.backward()
        gradient = entity_to_explain_embedding.grad[0].detach().cpu().numpy()
        entity_to_explain_embedding.grad = None         # reset the gradient, just to be sure

        # for each sample to add, compute the relevance by projection
        sample_2_relevance_by_projection = {}
        sample_2_gradient = {}
        for i in range(samples_to_add.shape[0]):
            current_sample = samples_to_add[i]

            cur_head_embedding = self.model.entity_embeddings[current_sample[0]].detach().reshape(1, self.model.dimension)
            cur_relation_embedding = self.model.relation_embeddings[current_sample[1]].detach().reshape(1, self.model.dimension)
            cur_tail_embedding = self.model.entity_embeddings[current_sample[2]].detach().reshape(1, self.model.dimension)
            cur_entity_to_explain_embedding = cur_head_embedding if perspective == "head" else cur_tail_embedding
            cur_entity_to_explain_embedding.requires_grad = True

            current_score = self.model.score_embeddings(cur_head_embedding, cur_relation_embedding, cur_tail_embedding)
            current_score.backward()
            current_gradient = cur_entity_to_explain_embedding.grad[0]
            projection_size = self.projection_size(gradient, current_gradient.cpu().numpy())

            sample_2_gradient[tuple(current_sample)] = current_gradient
            sample_2_relevance_by_projection[tuple(current_sample)] = projection_size

            cur_entity_to_explain_embedding.grad = None     # reset the gradient, just to be sure

        most_relevant_samples_by_projection = sorted(sample_2_relevance_by_projection.items(), key=lambda x:x[1], reverse=True)

        samples = numpy.array([(head_id, relation_id, tail_id) for _ in range(len(most_relevant_samples_by_projection))])


        # for each training sample containing the entity to explain,
        # shift the entity to explain in the gradient direction and verify the score improvement
        head_embeddings_tensor = self.model.entity_embeddings[samples[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples[:, 2]]
        for i in range(len(most_relevant_samples_by_projection)):
            item = most_relevant_samples_by_projection[i]
            current_sample = item[0]
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

    def projection_size(self, a: numpy.array, b: numpy.array):
        """
            Compute the size of the projection of b on the direction of a
            :param a:
            :param b:
            :return:
        """
        return numpy.dot(a, b) / numpy.linalg.norm(a)
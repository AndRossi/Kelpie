from typing import Tuple, Any

import numpy
from dataset import Dataset
from engines.engine import ExplanationEngine
from link_prediction.models.tucker import TuckER
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

        # compute the gradient of the fact to explain with respect to the embedding of the entity to explain.
        # the gradient points towards the direction that INCREASES the score.
        gradient = self.compute_gradient_for(sample=sample_to_explain, entity_to_explain=entity_to_explain_id)

        # move the embedding of the entity to explain in the direction that worsens the score
        # (that is, the direction that makes it greater if the model minimizes the scores of true facts,
        # or the direction that makes it smaller if the model aims at maximizes the scores of true facts)
        if self.model.is_minimizer():
            perturbed_entity_to_explain_embedding = self.model.entity_embeddings[entity_to_explain_id].detach() + self.epsilon * gradient.detach()
        else:
            perturbed_entity_to_explain_embedding = self.model.entity_embeddings[entity_to_explain_id].detach() - self.epsilon * gradient.detach()

        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in samples_containing_entity_to_explain])

        # compute the original scores of all facts containing the perspective entity
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]
        original_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()

        # replace the embedding of the perspective entity with the perturbed embedding everywhere, and recompute all scores
        for i, cur_sample in enumerate(samples_containing_entity_to_explain):
            if cur_sample[0] == entity_to_explain_id:
                head_embeddings_tensor[i] = perturbed_entity_to_explain_embedding
            else:
                assert(cur_sample[2] == entity_to_explain_id)
                tail_embeddings_tensor[i] = perturbed_entity_to_explain_embedding
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()

        # For each training sample featuring the perspective entity we now have both the original and the perturbed score.
        # Relevance = original score - lambda * perturbed score
        # (that is: we have tweaked a little bit the embedding of the perspective entity in order to worsen the fact to explain:
        # the more a training fact gets worsened as well, the more relevant it was.)
        sample_2_relevance = {}

        if self.model.is_minimizer():
            for i in range(len(samples_containing_entity_to_explain)):
                sample_2_relevance[tuple(samples_containing_entity_to_explain[i])] = - original_scores[i] + self.lambd * perturbed_scores[i]
        else:
            for i in range(len(samples_containing_entity_to_explain)):
                sample_2_relevance[tuple(samples_containing_entity_to_explain[i])] = original_scores[i] - self.lambd * perturbed_scores[i]

        # very important note: For the way in which we have designed relevance in both minimization and maximization case,
        # a high values always corresponds to (what we believe to be) big relevance
        most_relevant_samples = sorted(sample_2_relevance.items(), key=lambda x:x[1], reverse=True)

        if top_k == -1 or top_k < len(most_relevant_samples):
            return most_relevant_samples
        else:
            return most_relevant_samples[:top_k]


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

        target_entity_score, best_entity_score = self._target_and_best_scores_for_sample(numpy.array(sample_to_convert))

        # this is EXTREMELY important in models with dropout and/or batch normalization.
        # It disables dropout, and tells batch_norm layers to use saved statistics instead of batch data.
        # This affects both the computed scores and the computed gradients.
        self.model.eval()

        # compute the gradient of the fact to explain with respect to the embedding of the entity to explain
        gradient = self.compute_gradient_for(sample=sample_to_convert, entity_to_explain=entity_to_explain_id)

        # Shift the embedding of the entity to explain in the direction that improves the plausibility of the fact to explain.
        # If the model is a minimizer, this means a shift in opposite direction to the score gradient
        if self.model.is_minimizer():
            perturbed_entity_to_explain_embedding = self.model.entity_embeddings[entity_to_explain_id].detach() - self.epsilon * gradient.detach()
        # If the model is a maximizer, this means going in the same direction as the score gradient
        else:
            perturbed_entity_to_explain_embedding = self.model.entity_embeddings[entity_to_explain_id].detach() + self.epsilon * gradient.detach()

        samples_numpy_array = numpy.array([(head_id, relation_id, tail_id) for _ in samples_to_add])

        # compute the original scores of all the facts to add
        head_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[samples_numpy_array[:, 1]]
        tail_embeddings_tensor = self.model.entity_embeddings[samples_numpy_array[:, 2]]
        original_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()

        # replace the embedding of the perspective entity with the perturbed embedding everywhere, and recompute all scores
        for i in range(len(samples_to_add)):
            cur_sample_containing_entity_to_explain = samples_to_add[i]
            if cur_sample_containing_entity_to_explain[0] == entity_to_explain_id:
                head_embeddings_tensor[i] = perturbed_entity_to_explain_embedding
            else:
                assert(cur_sample_containing_entity_to_explain[2] == entity_to_explain_id)
                tail_embeddings_tensor[i] = perturbed_entity_to_explain_embedding
        perturbed_scores = self.model.score_embeddings(head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor).detach().cpu().numpy()

        # map each added sample to the corresponding
        added_sample_2_perturbed_score = {}
        for i, cur_added_sample in enumerate(samples_to_add):
            current_perturbed_score = perturbed_scores[i]
            added_sample_2_perturbed_score[cur_added_sample] = current_perturbed_score


        # For each sample to add, we now have both the original score and the score computed with the perturbed embedding.
        # We now compute the relevance of the samples to add.
        sample_2_relevance = {}
        for i, cur_added_sample in enumerate(samples_to_add):
            # if the model is a minimizer (= smaller scores correspond to more plausible facts),
            # the relevance corresponds to how much smaller the perturbed score has become than the original score.
            if self.model.is_minimizer():
                sample_2_relevance[tuple(cur_added_sample)] = original_scores[i] - self.lambd * perturbed_scores[i]

            # if the model is a maximizer (= greater scores correspond to more plausible facts),
            # the relevance corresponds to how much greater the perturbed score has become than the original score.
            else:
                sample_2_relevance[tuple(cur_added_sample)] = - original_scores[i] + self.lambd * perturbed_scores[i]

        # very important note: For the way in which we have designed relevance in both minimization and maximization case,
        # a high values always corresponds to (what we believe to be) big relevance
        sorted_samples_with_relevance = sorted(sample_2_relevance.items(), key=lambda x:x[1], reverse=True)

        return sorted_samples_with_relevance, \
               added_sample_2_perturbed_score, \
               target_entity_score, \
               best_entity_score


    def compute_gradient_for(self,
                             sample: Tuple[Any, Any, Any],
                             entity_to_explain: int):

        sample_head, sample_relation, sample_tail = sample

        entity_dimension = self.model.entity_dimension if isinstance(self.model, TuckER) else self.model.dimension
        relation_dimension = self.model.relation_dimension if isinstance(self.model, TuckER) else self.model.dimension

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


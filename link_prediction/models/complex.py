from typing import Tuple, Any
import numpy as np
import torch
from torch.nn import Parameter

from dataset import Dataset
from kelpie_dataset import KelpieDataset
from link_prediction.models.model import Model, KelpieModel, DIMENSION, INIT_SCALE


class ComplEx(Model):
    """
        The ComplEx class provides a Model implementation in PyTorch for the ComplEx system.
        This implementation adheres to paper "Canonical Tensor Decomposition for Knowledge Base Completion",
        and is largely inspired by the official implementation provided by the authors Lacroix, Usunier and Obozinski
        at https://github.com/facebookresearch/kbc/tree/master/kbc.

        In training or evaluation, our ComplEx class requires samples to be passed as 2-dimensional np.arrays.
        Each row corresponds to a sample and contains the integer ids of its head, relation and tail.
        Only *direct* samples should be passed to the model.

        TODO: add documentation about inverse facts and relations
        TODO: explain that the input must always be direct facts only
    """

    def __init__(self,
                 dataset: Dataset,
                 hyperparameters: dict,
                 init_random=True):
        """
            Constructor for ComplEx model.

            :param dataset: the Dataset on which to train and evaluate the model
            :param hyperparameters: hyperparameters dictionary.
                                    Must contain at least DIMENSION and INIT_SCALE
        """

        # note: the init_random parameter is important because when initializing a KelpieComplEx,
        #       self.entity_embeddings and self.relation_embeddings must not be initialized as Parameters!

        # initialize this object both as a Model and as a nn.Module
        Model.__init__(self, dataset)

        self.name = "ComplEx"
        self.dataset = dataset
        self.num_entities = dataset.num_entities  # number of entities in dataset
        self.num_relations = dataset.num_relations  # number of relations in dataset
        self.dimension = 2 * hyperparameters[
            DIMENSION]  # embedding dimension; it is 2* the passed dimension because each element must have both a real and an imaginary value
        self.real_dimension = hyperparameters[DIMENSION]  # dimension of the real part of each embedding

        self.init_scale = hyperparameters[INIT_SCALE]

        # create the embeddings for entities and relations as Parameters.
        # We do not use the torch.Embeddings module here in order to keep the code uniform to the KelpieComplEx model,
        # (on which torch.Embeddings can not be used as they do not allow the post-training mechanism).
        # We have verified that this does not affect performances in any way.
        # Each entity embedding and relation embedding is instantiated with size 2 * dimension because
        # for each entity and relation we store real and imaginary parts as separate elements
        if init_random:
            self.entity_embeddings = Parameter(torch.rand(self.num_entities, self.dimension).cuda(), requires_grad=True)
            self.relation_embeddings = Parameter(torch.rand(self.num_relations, self.dimension).cuda(),
                                                 requires_grad=True)

            # initialization step to make the embedding values smaller in the embedding space
            with torch.no_grad():
                self.entity_embeddings *= self.init_scale
                self.relation_embeddings *= self.init_scale

    def is_minimizer(self):
        """
        This method specifies whether this model aims at minimizing of maximizing scores .
        :return: True if in this model low scores are better than high scores; False otherwise.
        """
        return False

    def score(self, samples: np.array) -> np.array:
        """
            Compute scores for the passed samples
            :param samples: a 2-dimensional numpy array containing the samples to score, one per row
            :return: a numpy array containing the scores of the passed samples
        """
        lhs = self.entity_embeddings[samples[:, 0]]  # list of entity embeddings for the heads of the facts
        rel = self.relation_embeddings[samples[:, 1]]  # list of relation embeddings for the relations of the heads
        rhs = self.entity_embeddings[samples[:, 2]]  # list of entity embeddings for the tails of the facts

        return self.score_embeddings(lhs, rel, rhs).detach().cpu().numpy()

    def score_embeddings(self,
                         head_embeddings: torch.Tensor,
                         rel_embeddings: torch.Tensor,
                         tail_embeddings: torch.Tensor):
        """
            Compute scores for the passed triples of head, relation and tail embeddings.
            :param head_embeddings: a torch.Tensor containing the embeddings representing the head entities
            :param rel_embeddings: a torch.Tensor containing the embeddings representing the relations
            :param tail_embeddings: a torch.Tensor containing the embeddings representing the tail entities

            :return: a numpy array containing the scores computed for the passed triples of embeddings
        """

        # NOTE: this method is extremely important, because apart from being called by the ComplEx score(samples) method
        # it is also used to perform several operations on gradients in our GradientEngine
        # as well as the operations from the paper "Data Poisoning Attack against Knowledge Graph Embedding"
        # that we use as a baseline and as a heuristic for our work.

        # split the head embedding into real and imaginary components
        lhs = head_embeddings[:, :self.real_dimension], head_embeddings[:, self.real_dimension:]
        # split the relation embedding into real and imaginary components
        rel = rel_embeddings[:, :self.real_dimension], rel_embeddings[:, self.real_dimension:]
        # split the tail embedding into real and imaginary components
        rhs = tail_embeddings[:, :self.real_dimension], tail_embeddings[:, self.real_dimension:]

        # Bilinear product lhs * rel * rhs in complex scenario:
        #   lhs => lhs[0] + i*lhs[1]
        #   rel => rel[0] + i*rel[1]
        #   rhs => rhs[0] - i*rhs[1] (complex conjugate of the tail: negative sign to its imaginary part)
        #
        #   ( lhs[0] + i*lhs[1] )    *    ( rel[0] + i*rel[1] )    *    ( rhs[0] - i*rhs[1] )
        #   ( lhs[0]*rel[0] + i*(lhs[0]*rel[1]) + i*(lhs[1]*rel[0]) + i*i*(lhs[1]*rel[1] )    *    ( rhs[0] - i*rhs[1] )

        #   ( lhs[0]*rel[0] - lhs[1]*rel[1] + i(lhs[0]*rel[1] + lhs[1]*rel[0]) )    *    ( rhs[0] - i*rhs[1] )
        #
        #   (lhs[0]*rel[0] - lhs[1]*rel[1]) * rhs[0]  + i*(...)*rhs[0] - i*(...)*rhs[1] - i*i*(lhs[0]*rel[1] + lhs[1]*rel[0])*rhs[1]
        #   (lhs[0]*rel[0] - lhs[1]*rel[1]) * rhs[0] + (lhs[0]*rel[1] + lhs[1]*rel[0]) * rhs[1] + i*(...)
        #
        # => the real part of the result is:
        #    (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
        #    (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1]
        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def all_scores(self, samples: np.array) -> np.array:
        """
            For each of the passed samples, compute scores for all possible tail entities.
            :param samples: a 2-dimensional numpy array containing the samples to score, one per row
            :return: a 2-dimensional numpy array that, for each sample, contains a row for each passed sample
                     and a column for each possible tail
        """

        # for each fact <cur_head, cur_rel, cur_tail> to predict, get all (cur_head, cur_rel) couples
        # and compute the scores using any possible entity as a tail
        q = self._get_queries(samples)
        rhs = self._get_rhs()
        return q @ rhs  # 2d matrix: each row corresponds to a sample and has the scores for all entities

    def forward(self, samples: np.array):
        """
            Perform forward propagation on the passed samples
            :param samples: a 2-dimensional numpy array containing the samples to use in forward propagation, one per row
            :return: a tuple containing
                        - the scores for each passed sample with all possible tails
                        - a partial result to use in regularization
        """

        lhs = self.entity_embeddings[samples[:, 0]]  # list of entity embeddings for the heads of the facts
        rel = self.relation_embeddings[samples[:, 1]]  # list of relation embeddings for the relations of the heads
        rhs = self.entity_embeddings[samples[:, 2]]  # list of entity embeddings for the tails of the facts

        lhs = lhs[:, :self.real_dimension], lhs[:, self.real_dimension:]
        rel = rel[:, :self.real_dimension], rel[:, self.real_dimension:]
        rhs = rhs[:, :self.real_dimension], rhs[:, self.real_dimension:]

        to_score = self.entity_embeddings
        to_score = to_score[:, :self.real_dimension], to_score[:, self.real_dimension:]

        # this returns two factors
        #   factor 1 is one matrix with the scores for the fact
        #
        #   factor 2 is 3 matrices that will then be used for regularization:
        #           matrix 1: for each head entity get the real and imaginary component,
        #                           square each of their elements
        #                           sum the resulting vectors
        #                           squareroot the elements of the resulting vector
        #           matrix 2: for each relation get the real and imaginary component,
        #                           square each of their elements
        #                           sum the resulting vectors
        #                           squareroot the elements of the resulting vector
        #           matrix 3: for each tail entity get the real and imaginary component,
        #                           square each of their elements
        #                           sum the resulting vectors
        #                           squareroot the elements of the resulting vector
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               )

    def _get_rhs(self):
        """
            This private method computes, for each fact to score,
            a partial factor of the score that only depends on the tail of the fact.
            In our scenario, this actually just corresponds to the transposed embeddings of all entities.

            Note: This method is used in conjunction with _get_queries
            to obtain very efficiently, for each test fact, the score for each possible tail:
                scores = output of _get_queries @ output of _get_rhs

            :return: a torch.Tensor containing the embeddings of all entities (one per column)
        """
        return self.entity_embeddings[:self.dataset.num_entities].transpose(0, 1).detach()

    def _get_queries(self, samples: np.array):
        """
            This private method computes, for each fact to score,
            the partial factor of the score that only depends on the head and relation of the fact

            Note: This method is used in conjunction with _get_rhs
            to obtain very efficiently, for each test fact, the score for each possible tail:
                    scores = output of get_queries @ output of get_rhs

            :param samples: the facts to compute the partial factors for, as a np.array
            :return: a torch.Tensor with, in each row, the partial score factor of the head and relation for one fact
        """

        # get the embeddings for head and relation of each fact
        head_embeddings = self.entity_embeddings[samples[:, 0]]
        relation_embeddings = self.relation_embeddings[samples[:, 1]]

        # split both head embeddings and relation embeddings into real and imaginary parts
        head_embeddings = head_embeddings[:, :self.real_dimension], head_embeddings[:, self.real_dimension:]
        relation_embeddings = relation_embeddings[:, :self.real_dimension], relation_embeddings[:, self.real_dimension:]

        # obtain a tensor that, in each row, has the partial score factor of the head and relation for one fact
        return torch.cat([
            head_embeddings[0] * relation_embeddings[0] - head_embeddings[1] * relation_embeddings[1],
            head_embeddings[0] * relation_embeddings[1] + head_embeddings[1] * relation_embeddings[0]
        ], 1)

    def predict_samples(self, samples: np.array) -> Tuple[Any, Any, Any]:
        """
            This method takes as an input a tensor of 'direct' samples,
            runs head and tail prediction on each of them
            and returns
                - the obtained scores for direct and inverse version of each sample,
                - the obtained head and tail ranks for each sample
                - the list of predicted entities for each sample
            :param samples: a torch.Tensor containing all the DIRECT samples to predict.
                            They will be automatically inverted to perform head prediction

            :return: three dicts mapping each passed direct sample (in Tuple format) respectively to
                        - the scores of that direct sample and of the corresponding inverse sample;
                        - the head and tail rank for that sample;
                        - the head and tail predictions for that sample
        """

        direct_samples = samples

        # make sure that all the passed samples are "direct" samples
        assert np.all(direct_samples[:, 1] < self.dataset.num_direct_relations)

        # output data structures
        scores, ranks, predictions = [], [], []

        # invert samples to perform head predictions
        inverse_samples = self.dataset.invert_samples(direct_samples)

        # obtain scores, ranks and predictions both for direct and inverse samples
        direct_scores, tail_ranks, tail_predictions = self.predict_tails(direct_samples)
        inverse_scores, head_ranks, head_predictions = self.predict_tails(inverse_samples)

        for i in range(direct_samples.shape[0]):
            # add to the scores list a couple containing the scores of the direct and of the inverse sample
            scores.append((direct_scores[i], inverse_scores[i]))

            # add to the ranks list a couple containing the ranks of the head and of the tail
            ranks.append((int(head_ranks[i]), int(tail_ranks[i])))

            # add to the prediction list a couple containing the lists of predictions
            predictions.append((head_predictions[i], tail_predictions[i]))

        return scores, ranks, predictions

    def predict_tails(self, samples: np.array) -> Tuple[Any, Any, Any]:
        """
            Returns filtered scores, ranks and predicted entities for each passed fact.
            :param samples: a torch.LongTensor of triples (head, relation, tail).
                          The triples can also be "inverse triples" with (tail, inverse_relation_id, head)
            :return:
        """

        ranks = torch.ones(len(samples))  # initialize with ONES

        with torch.no_grad():

            # compute scores for each sample for all possible tails
            all_scores = self.all_scores(samples)

            # from the obtained scores, extract the the scores of the actual facts <cur_head, cur_rel, cur_tail>
            targets = torch.zeros(size=(len(samples), 1)).cuda()
            for i, (_, _, tail_id) in enumerate(samples):
                targets[i, 0] = all_scores[i, tail_id].item()

            # set to -1e6 the scores obtained using tail entities that must be filtered out (filtered scenario)
            # In this way, those entities will be ignored in rankings
            for i, (head_id, rel_id, tail_id) in enumerate(samples):
                # get the list of tails to filter out; include the actual target tail entity too
                filter_out = self.dataset.to_filter[(head_id, rel_id)]

                if tail_id not in filter_out:
                    filter_out.append(tail_id)

                all_scores[i, torch.LongTensor(filter_out)] = -1e6

            # fill the ranks data structure and convert it to a Python list
            ranks += torch.sum((all_scores >= targets).float(), dim=1).cpu()  # ranks was initialized with ONES
            ranks = ranks.cpu().numpy().tolist()

            all_scores = all_scores.cpu().numpy()
            targets = targets.cpu().numpy()

            # save the list of all obtained scores
            scores = [targets[i, 0] for i in range(len(samples))]

            predictions = []
            for i, (head_id, rel_id, tail_id) in enumerate(samples):
                filter_out = self.dataset.to_filter[(head_id, rel_id)]
                if tail_id not in filter_out:
                    filter_out.append(tail_id)

                predicted_tails = np.where(all_scores[i] > -1e6)[0]

                # get all not filtered tails and corresponding scores for current fact
                # predicted_tails = np.where(all_scores[i] != -1e6)
                predicted_tails_scores = all_scores[i, predicted_tails]  # for cur_tail in predicted_tails]

                # note: the target tail score and the tail id are in the same position in their respective lists!
                # predicted_tails_scores = np.append(predicted_tails_scores, scores[i])
                # predicted_tails = np.append(predicted_tails, [tail_id])

                # sort the scores and predicted tails list in the same way
                permutation = np.argsort(-predicted_tails_scores)

                predicted_tails_scores = predicted_tails_scores[permutation]
                predicted_tails = predicted_tails[permutation]

                # include the score of the target tail in the predictions list
                # after ALL entities with greater or equal scores (MIN policy)
                j = 0
                while j < len(predicted_tails_scores) and predicted_tails_scores[j] >= scores[i]:
                    j += 1

                predicted_tails_scores = np.concatenate((predicted_tails_scores[:j],
                                                         np.array([scores[i]]),
                                                         predicted_tails_scores[j:]))
                predicted_tails = np.concatenate((predicted_tails[:j],
                                                  np.array([tail_id]),
                                                  predicted_tails[j:]))

                # add to the results data structure
                predictions.append(predicted_tails)  # as a np array!

        return scores, ranks, predictions

    def criage_first_step(self, samples: np.array):
        return self._get_queries(samples)

    def criage_last_step(self,
                         x: torch.Tensor,
                         tail_embeddings: torch.Tensor):
        return x @ tail_embeddings

    def kelpie_model_class(self):
        return KelpieComplEx


################

class KelpieComplEx(KelpieModel, ComplEx):
    def __init__(
            self,
            dataset: KelpieDataset,
            model: ComplEx,
            init_tensor=None):

        ComplEx.__init__(self,
                         dataset=dataset,
                         hyperparameters={DIMENSION: model.real_dimension,
                                          INIT_SCALE: model.init_scale},
                         init_random=False)

        self.model = model
        self.original_entity_id = dataset.original_entity_id
        self.kelpie_entity_id = dataset.kelpie_entity_id

        # extract the values of the trained embeddings for entities and relations and freeze them.
        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        frozen_relation_embeddings = model.relation_embeddings.clone().detach()

        # the tensor from which to initialize the kelpie_entity_embedding;
        # if it is None it is initialized randomly
        if init_tensor is None:
            init_tensor = torch.rand(1, self.dimension)

        # It is *extremely* important that kelpie_entity_embedding is both a Parameter and an instance variable
        # because the whole approach of the project is to obtain the parameters model params with parameters() method
        # and to pass them to the Optimizer for optimization.
        #
        # If I used .cuda() outside the Parameter, like
        #       self.kelpie_entity_embedding = Parameter(torch.rand(1, self.dimension), requires_grad=True).cuda()
        # IT WOULD NOT WORK because cuda() returns a Tensor, not a Parameter.

        # Therefore kelpie_entity_embedding would not be a Parameter anymore.
        self.kelpie_entity_embedding = Parameter(init_tensor.cuda(), requires_grad=True)

        with torch.no_grad():  # Initialize as any other embedding
            self.kelpie_entity_embedding *= self.init_scale

        self.relation_embeddings = frozen_relation_embeddings
        self.entity_embeddings = torch.cat([frozen_entity_embeddings, self.kelpie_entity_embedding], 0)

    # Override
    def predict_samples(self,
                        samples: np.array,
                        original_mode: bool = False):
        """
        This method overrides the Model predict_samples method
        by adding the possibility to run predictions in original_mode,
        which means ignoring in any circumstances the additional "fake" kelpie entity.

        :param samples: the DIRECT samples. Will be inverted to perform head prediction
        :param original_mode:

        :return:
        """

        direct_samples = samples

        # assert all samples are direct
        assert (samples[:, 1] < self.dataset.num_direct_relations).all()

        # if we are in original_mode, make sure that the kelpie entity is not featured in the samples to predict
        # otherwise, make sure that the original entity is not featured in the samples to predict
        forbidden_entity_id = self.kelpie_entity_id if original_mode else self.original_entity_id
        assert np.isin(forbidden_entity_id, direct_samples[:][0, 2]) == False

        # use the Model implementation method to obtain scores, ranks and prediction results.
        # these WILL feature the forbidden entity, so we now need to filter them
        scores, ranks, predictions = ComplEx.predict_samples(self, direct_samples)

        # remove any reference to the forbidden entity id
        # (that may have been included in the predicted entities)
        for i in range(len(direct_samples)):
            head_predictions, tail_predictions = predictions[i]
            head_rank, tail_rank = ranks[i]

            # remove the forbidden entity id from the head predictions (note: it could be absent due to filtering)
            # and if it was before the head target decrease the head rank by 1
            forbidden_indices = np.where(head_predictions == forbidden_entity_id)[0]
            if len(forbidden_indices) > 0:
                index = forbidden_indices[0]
                head_predictions = np.concatenate([head_predictions[:index], head_predictions[index + 1:]], axis=0)
                if index < head_rank:
                    head_rank -= 1

            # remove the kelpie entity id from the tail predictions  (note: it could be absent due to filtering)
            # and if it was before the tail target decrease the head rank by 1
            forbidden_indices = np.where(tail_predictions == forbidden_entity_id)[0]
            if len(forbidden_indices) > 0:
                index = forbidden_indices[0]
                tail_predictions = np.concatenate([tail_predictions[:index], tail_predictions[index + 1:]], axis=0)
                if index < tail_rank:
                    tail_rank -= 1

            predictions[i] = (head_predictions, tail_predictions)
            ranks[i] = (head_rank, tail_rank)

        return scores, ranks, predictions

    # Override
    def predict_sample(self,
                       sample: np.array,
                       original_mode: bool = False):
        """
        Override the
        :param sample: the DIRECT sample. Will be inverted to perform head prediction
        :param original_mode:
        :return:
        """

        assert sample[1] < self.dataset.num_direct_relations

        scores, ranks, predictions = self.predict_samples(np.array([sample]), original_mode)
        return scores[0], ranks[0], predictions[0]

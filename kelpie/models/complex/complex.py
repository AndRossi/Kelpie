from abc import ABC
import torch
from torch import nn
import numpy as np
from torch.nn import Parameter

from kelpie.models.complex.dataset import ComplExDataset, KelpieComplExDataset


class ComplEx(nn.Module, ABC):
    """
        This class provides the implementation for the ComplEx model.
        The implementation follows paper "Canonical Tensor Decomposition for Knowledge Base Completion",
        and is largely inspired by the official implementation provided by the authors Lacroix, Usunier and Obozinski
        at https://github.com/facebookresearch/kbc/tree/master/kbc
    """

    # (takes in input np.arrays and, when necessary, converts them into torch.tensors)

    def __init__(
            self,
            dataset: ComplExDataset,
            dimension: int,
            init_random = True,
            init_size: float = 1e-3
    ):
        """
            Constructor for ComplEx model.

            :param dataset: the ComplExDataset on which to train and evaluate the model
            :param dimension: embedding dimension
            :param init_size: factor to use to make the embedding values smaller at initialization
        """

        # note: the init_random parameter is important because when initializing a KelpieComplEx,
        #       self.entity_embeddings and self.relation_embeddings must not be initialized as Parameters!

        super(ComplEx, self).__init__()

        self.dataset = dataset                       # ComplExDataset on which the model will be trained and evaluated
        self.num_entities = dataset.num_entities     # number of entities in dataset
        self.num_relations = dataset.num_relations   # number of relations in dataset
        self.dimension = dimension                   # embedding dimension

        # create the embeddings for entities and relations as Parameters.
        # We do not use the torch.Embeddings module here in order to keep the code uniform to the KelpieComplEx model,
        # (on which torch.Embeddings can not be used as they do not allow the post-training mechanism).
        # We have verified that this does not affect performances in any way.
        # Each entity embedding and relation embedding is instantiated with size 2 * dimension because
        # for each entity and relation we store real and imaginary parts as separate elements
        if init_random:
            self.entity_embeddings = Parameter(torch.rand(self.num_entities, 2 * self.dimension).cuda(), requires_grad=True)
            self.relation_embeddings = Parameter(torch.rand(self.num_relations, 2 * self.dimension).cuda(), requires_grad=True)

            # initialization step to make the embedding values smaller in the embedding space
            with torch.no_grad():
                self.entity_embeddings *= init_size
                self.relation_embeddings *= init_size


    def score(self, samples: np.array):
        """

        :param samples:
        :return:
        """
        lhs = self.entity_embeddings[samples[:, 0]]      # list of entity embeddings for the heads of the facts
        rel = self.relation_embeddings[samples[:, 1]]    # list of relation embeddings for the relations of the heads
        rhs = self.entity_embeddings[samples[:, 2]]      # list of entity embeddings for the tails of the facts

        lhs = lhs[:, :self.dimension], lhs[:, self.dimension:]   # split head embeddings into real and imaginary components
        rel = rel[:, :self.dimension], rel[:, self.dimension:]   # split relation embeddings into real and imaginary components
        rhs = rhs[:, :self.dimension], rhs[:, self.dimension:]   # split tail embeddings into real and imaginary components

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
        # WHY NOT USING THE IMAGINARY PART TOO?
        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, samples: np.array):
        lhs = self.entity_embeddings[samples[:, 0]]       # list of entity embeddings for the heads of the facts
        rel = self.relation_embeddings[samples[:, 1]]     # list of relation embeddings for the relations of the heads
        rhs = self.entity_embeddings[samples[:, 2]]       # list of entity embeddings for the tails of the facts

        lhs = lhs[:, :self.dimension], lhs[:, self.dimension:]
        rel = rel[:, :self.dimension], rel[:, self.dimension:]
        rhs = rhs[:, :self.dimension], rhs[:, self.dimension:]

        to_score = self.entity_embeddings
        to_score = to_score[:, :self.dimension], to_score[:, self.dimension:]

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
        head_embeddings = head_embeddings[:, :self.dimension], head_embeddings[:, self.dimension:]
        relation_embeddings = relation_embeddings[:, :self.dimension], relation_embeddings[:, self.dimension:]

        # obtain a tensor that, in each row, has the partial score factor of the head and relation for one fact
        return torch.cat([
            head_embeddings[0] * relation_embeddings[0] - head_embeddings[1] * relation_embeddings[1],
            head_embeddings[0] * relation_embeddings[1] + head_embeddings[1] * relation_embeddings[0]
        ], 1)

    def predict_sample(self, direct_sample: np.array):
        sample_tuple = (direct_sample[0], direct_sample[1], direct_sample[2])
        sample_2_scores, sample_2_ranks, sample_2_predictions = self.predict_samples(np.array([direct_sample]))

        scores = sample_2_scores[sample_tuple]
        ranks = sample_2_ranks[sample_tuple]
        predictions = sample_2_predictions[sample_tuple]

        return scores, ranks, predictions

    def predict_samples(self, direct_samples: np.array):
        """
            This method takes as an input a tensor of 'direct' samples,
            runs head and tail prediction on each of them
            and returns
                - the obtained scores for direct and inverse version of each sample,
                - the obtained head and tail ranks for each sample
                - the list of predicted entities for each sample
            :param direct_samples: a torch.Tensor containing all the samples to predict

            :return: three dicts mapping each passed direct sample (in Tuple format) respectively to
                        - the scores of that direct sample and of the corresponding inverse sample;
                        - the head and tail rank for that sample;
                        - the head and tail predictions for that sample (up to the target entities)
        """

        # make sure that all the passed samples are "direct" samples
        assert np.all(direct_samples[:, 1] < self.dataset.num_original_relations)

        # output data structures
        sample_2_out_scores, sample_2_out_ranks, sample_2_out_predictions = dict(), dict(), dict()

        # invert samples to perform head predictions
        inverse_samples = self.dataset.invert_samples(direct_samples)

        #obtain scores, ranks and predictions both for direct and inverse samples
        direct_sample_2_score, direct_sample_2_rank, direct_sample_2_predictions = self.predict_tails(direct_samples)
        inverse_sample_2_score, inverse_sample_2_rank, inverse_sample_2_predictions = self.predict_tails(inverse_samples)

        for i in range(direct_samples.shape[0]):
            h, r, t = direct_samples[i]
            i_h, i_r, i_t = inverse_samples[i]

            # map each (direct) sample to a couple containing both the score of the direct and of the inverse sample
            sample_2_out_scores[(h, r, t)] = (direct_sample_2_score[(h, r, t)], inverse_sample_2_score[(i_h, i_r, i_t)])
            # map each (direct) sample to a couple containing both the rank of the head and of the tail
            sample_2_out_ranks[(h, r, t)] = (inverse_sample_2_rank[(i_h, i_r, i_t)], direct_sample_2_rank[(h, r, t)], )
            # map each (direct) sample to a couple containing both the predictions for the head and for the tail
            sample_2_out_predictions[(h, r, t)] = (inverse_sample_2_predictions[(i_h, i_r, i_t)], direct_sample_2_predictions[(h, r, t)])

        return sample_2_out_scores, sample_2_out_ranks, sample_2_out_predictions


    def predict_tails(self, samples: np.array):
        """
        Returns filtered scores, ranks and predicted entities for each passed fact.
        :param samples: a torch.LongTensor of triples (head, relation, tail).
                      The triples can also be "inverse triples" with (tail, inverse_relation_id, head)
        :return:
        """

        ranks = torch.ones(len(samples))    # initialize with ONES
        sample_2_score, sample_2_rank, sample_2_predictions = dict(), dict(), dict()  # output data structures

        with torch.no_grad():

            # for each fact <cur_head, cur_rel, cur_tail> to predict, get all (cur_head, cur_rel) couples
            # and compute the scores using any possible entity as a tail
            q = self._get_queries(samples)
            rhs = self._get_rhs()
            scores = q @ rhs

            # from the obtained scores, extract the the scores of the actual facts <cur_head, cur_rel, cur_tail>
            targets = torch.zeros(size=(len(samples), 1)).cuda()
            for i, (_, _, tail_id) in enumerate(samples):
                targets[i, 0] = scores[i, tail_id].item()

            # set to -1e6 the scores obtained using tail entities that must be filtered out (filtered scenario)
            # In this way, those entities will be ignored in rankings
            for i, (head_id, rel_id, tail_id) in enumerate(samples):
                # get the list of tails to filter out; include the actual target tail entity too
                filter_out = self.dataset.to_filter[(head_id, rel_id)]

                if tail_id not in filter_out:
                    filter_out.append(tail_id)

                scores[i, torch.LongTensor(filter_out)] = -1e6

            ranks += torch.sum((scores >= targets).float(), dim=1).cpu()    #ranks was initialized with ONES

            mistakes = (scores >= targets).cpu().numpy()

            scores = scores.cpu().numpy()
            ranks = ranks.cpu().numpy()
            for i, (head_id, rel_id, tail_id) in enumerate(samples):
                query_mistakes = mistakes[i]
                predicted_tails = np.array(np.where(query_mistakes==1))[0]

                if len(predicted_tails) > 0:
                    predicted_tails_scores = np.array([scores[i, cur_tail] for cur_tail in predicted_tails])
                    permutation = np.argsort(-predicted_tails_scores)
                    predicted_tails = predicted_tails[permutation]

                predicted_tails = predicted_tails.tolist()
                predicted_tails.append(tail_id)

                sample_2_predictions[(head_id, rel_id, tail_id)] = predicted_tails
                sample_2_rank[(head_id, rel_id, tail_id)] = ranks[i]
                sample_2_score[(head_id, rel_id, tail_id)] = targets[i, 0]

        return sample_2_score, sample_2_rank, sample_2_predictions


class KelpieComplEx(ComplEx):
    def __init__(
            self,
            dataset: KelpieComplExDataset,
            model: ComplEx,
            init_size: float = 1e-3
    ):

        super(KelpieComplEx, self).__init__(dataset=dataset,
                                            dimension=model.dimension,
                                            init_random=False,
                                            init_size=init_size)

        self.original_entity_id = dataset.original_entity_id
        self.kelpie_entity_id = dataset.kelpie_entity_id

        # extract the values of the trained embeddings for entities and relations and freeze them.
        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        frozen_relation_embeddings = model.relation_embeddings.clone().detach()

        # It is *extremely* important that entity_to_explain_embedding is both a Parameter and an instance variable
        # because the whole approach of the project is to obtain the parameters model params with parameters() method
        # and to pass them to the Optimizer for optimization.
        #
        # If I used .cuda() outside the Parameter, like
        #       self.entity_to_explain_embedding = Parameter(torch.rand(1, 2*self.dimension), requires_grad=True).cuda()
        # IT WOULD NOT WORK because cuda() returns a Tensor, not a Parameter.

        # Therefore entity_to_explain_embedding would not be a Parameter anymore.

        self.kelpie_entity_embedding = Parameter(torch.rand(1, 2*self.dimension).cuda(), requires_grad=True)
        with torch.no_grad():           # Initialize as any other embedding
            self.kelpie_entity_embedding *= init_size

        self.entity_embeddings = torch.cat([frozen_entity_embeddings, self.kelpie_entity_embedding], 0)
        self.relation_embeddings = frozen_relation_embeddings

    def update_embeddings(self):
        with torch.no_grad():
            self.entity_embeddings[self.kelpie_entity_id] = self.kelpie_entity_embedding
from typing import Tuple, Any

import torch
from torch import nn
import numpy as np
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

from dataset import Dataset
from kelpie_dataset import KelpieDataset
from link_prediction.models.model import Model, DIMENSION, KelpieModel


class TransE(Model):
    """
        The TransE class provides a Model implementation in PyTorch for the TransE system.
        We hardcode L1 norm in this implementation, because it has been observed to yield the best results.

        In training or evaluation, our TransE class requires samples to be passed as 2-dimensional np.arrays.
        Each row corresponds to a sample and contains the integer ids of its head, relation and tail.
        Only *direct* samples should be passed to the model.

        TODO: add documentation about inverse facts and relations
        TODO: explain that the input must always be direct facts only
    """


    def __init__(self,
                 dataset: Dataset,
                 hyperparameters: dict,
                 init_random = True):
        """
            Constructor for TransE model.

            :param dataset: the Dataset on which to train and evaluate the model
            :param hyperparameters: a dict containing the model hyperparameters. It must contain at least:
                        - dimension: embedding dimension
                        - init_scale: factor to use to make the embedding values smaller at initialization
        """

        # note: the init_random parameter is important because when initializing a KelpieComplEx,
        #       self.entity_embeddings and self.relation_embeddings must not be initialized as Parameters!

        # initialize this object both as a Model and as a nn.Module
        Model.__init__(self, dataset)
        nn.Module.__init__(self)

        self.args=dataset.args
        self.name = "TransE"
        self.dataset = dataset
        self.num_entities = dataset.num_entities     # number of entities in dataset
        self.num_relations = dataset.num_relations   # number of all relations in dataset (including the inverted ones)
        self.dimension = hyperparameters[DIMENSION]  # embedding dimension

        # create the embeddings for entities and relations as Parameters, using the passed dimension as size.
        # We do not use the torch.Embeddings module here in order to keep the code uniform to the KelpieDistMult model,
        # (on which torch.Embeddings can not be used as they do not allow the post-training mechanism).
        # We have verified that this does not affect performances in any way.
        if init_random:
            self.entity_embeddings = Parameter(torch.rand(self.num_entities, self.dimension).cuda(), requires_grad=True)
            self.relation_embeddings = Parameter(torch.rand(self.num_relations, self.dimension).cuda(), requires_grad=True)
            xavier_normal_(self.entity_embeddings)
            xavier_normal_(self.relation_embeddings)

    def is_minimizer(self):
        """
        This method specifies whether this model aims at minimizing of maximizing scores .
        :return: True if in this model low scores are better than high scores; False otherwise.
        """
        return True

    def score(self, samples: np.array) -> np.array:
        """
            Compute scores for the passed samples
            :param samples: a 2-dimensional numpy array containing the samples to score, one per row
            :return: a numpy array containing the scores of the passed samples
        """
        head_embeddings = self.entity_embeddings[samples[:, 0]]     # list of entity embeddings for the heads of the facts
        rel_embeddings = self.relation_embeddings[samples[:, 1]]    # list of relation embeddings for the relations of the heads
        tail_embeddings = self.entity_embeddings[samples[:, 2]]     # list of entity embeddings for the tails of the facts

        return self.score_embeddings(head_embeddings, rel_embeddings, tail_embeddings).detach().cpu().numpy()

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
        # NOTE: this method is extremely important, because apart from being called by the Transe score(samples) method
        # it is also used to perform the operations of paper "Data Poisoning Attack against Knowledge Graph Embedding"
        # that we use as a baseline for our work.
        return (head_embeddings + rel_embeddings - tail_embeddings).norm(p=1, dim=1)

    def all_scores(self, samples: np.array):
        """
            For each of the passed samples, compute scores for all possible tail entities.

            :param samples: a 2-dimensional numpy array containing the samples to score, one per row
            :return: a 2-dimensional numpy array with a row for each passed sample and a column for each possible tail;
                     in any position, it contains the score for the sample of that row, computed with the tail of that column
        """

        head_embeddings = self.entity_embeddings[samples[:, 0]]     # list of entity embeddings for the heads of the facts
        rel_embeddings = self.relation_embeddings[samples[:, 1]]    # list of relation embeddings for the relations of the fats
        all_tail_embeddings = self.entity_embeddings                # the list of tails to use is the list of all entity embeddings
        translation_outputs = head_embeddings + rel_embeddings

        #all_scores = torch.stack([(translation_outputs-cur_tail_embedding).norm(p=1, dim=1) for cur_tail_embedding in all_tail_embeddings])
        all_scores = (translation_outputs.unsqueeze(0) - all_tail_embeddings.unsqueeze(1)).norm(p=1, dim=2)

        return all_scores.transpose(0, 1)

    def forward(self, samples: np.array):
        """
            Perform forward propagation on the passed samples
            :param samples: a 2-dimensional numpy array containing the samples to use in forward propagation, one per row
            :return: a tuple containing
                        - the scores for each passed sample with all possible tails
                        - a partial result to use in regularization
        """

        head_embeddings = self.entity_embeddings[samples[:, 0]]       # list of entity embeddings for the heads of the facts
        rel_embeddings = self.relation_embeddings[samples[:, 1]]     # list of relation embeddings for the relations of the heads
        tail_embeddings = self.entity_embeddings[samples[:, 2]]       # list of entity embeddings for the tails of the facts

        # this returns two factors
        #   factor 1 is one matrix with the scores for the facts
        #   factor 2 contains the absolute values of the head embeddings, relation embeddings and tail embeddings,
        #            to use for regularization:
        return (head_embeddings + rel_embeddings - tail_embeddings).norm(p=1, dim=1), \
               (head_embeddings, rel_embeddings, tail_embeddings)


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

        #obtain scores, ranks and predictions both for direct and inverse samples
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

        ranks = torch.ones(len(samples))    # initialize with ONES

        with torch.no_grad():

            # compute scores for each sample for all possible tails
            all_scores = self.all_scores(samples)

            # from the obtained scores, extract the the scores of the actual facts <cur_head, cur_rel, cur_tail>
            targets = torch.zeros(size=(len(samples), 1)).cuda()
            for i, (_, _, tail_id) in enumerate(samples):
                targets[i, 0] = all_scores[i, tail_id].item()

            # set to +1e6 the scores obtained using tail entities that must be filtered out (filtered scenario)
            # In this way, those entities will be ignored in rankings
            for i, (head_id, rel_id, tail_id) in enumerate(samples):
                # get the list of tails to filter out; include the actual target tail entity too
                filter_out = self.dataset.to_filter[(head_id, rel_id)]

                if tail_id not in filter_out:
                    filter_out.append(tail_id)

                all_scores[i, torch.LongTensor(filter_out)] = 1e6

            # fill the ranks data structure and convert it to a Python list
            ranks += torch.sum((all_scores <= targets).float(), dim=1).cpu()    #ranks was initialized with ONES
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

                predicted_tails = np.where(all_scores[i] < 1e6)[0]

                # get all not filtered tails and corresponding scores for current fact
                #predicted_tails = np.where(all_scores[i] != -1e6)
                predicted_tails_scores = all_scores[i, predicted_tails] #for cur_tail in predicted_tails]

                # note: the target tail score and the tail id are in the same position in their respective lists!
                #predicted_tails_scores = np.append(predicted_tails_scores, scores[i])
                #predicted_tails = np.append(predicted_tails, [tail_id])

                # sort the scores and predicted tails list in the same way
                permutation = np.argsort(predicted_tails_scores)

                predicted_tails_scores = predicted_tails_scores[permutation]
                predicted_tails = predicted_tails[permutation]

                # include the score of the target tail in the predictions list
                # after ALL entities with lesser or equal scores (MIN policy)
                j = 0
                while j < len(predicted_tails_scores) and predicted_tails_scores[j] <= scores[i]:
                    j += 1

                predicted_tails_scores = np.concatenate((predicted_tails_scores[:j],
                                                         np.array([scores[i]]),
                                                         predicted_tails_scores[j:]))
                predicted_tails = np.concatenate((predicted_tails[:j],
                                                  np.array([tail_id]),
                                                  predicted_tails[j:]))

                # add to the results data structure
                predictions.append(predicted_tails)     # as a np array!

        return scores, ranks, predictions

    def kelpie_model_class(self):
        return KelpieTransE

################

class KelpieTransE(KelpieModel, TransE):
    def __init__(
            self,
            dataset: KelpieDataset,
            model: TransE,
            init_tensor=None):

        TransE.__init__(self,
                        dataset=dataset,
                        hyperparameters={DIMENSION: model.dimension},
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
        with torch.no_grad():           # Initialize as any other embedding
            xavier_normal_(self.kelpie_entity_embedding)

        self.relation_embeddings = frozen_relation_embeddings
        self.entity_embeddings = torch.cat([frozen_entity_embeddings, self.kelpie_entity_embedding], 0)

    # Override
    def predict_samples(self,
                        samples: np.array,
                        original_mode: bool = False):
        """
        This method overrides the Model predict_samples method by adding the possibility to run predictions
        either in original_mode (i.e., ignoring the kelpie entity, or mimic, and using the original embedding instead)
        or not (i.e., using the post-trained embedding instead of the original one)

        :param samples: the DIRECT samples. Will be inverted to perform head prediction
        :param original_mode: a boolean flag specifying whether or not the predictions should be run in original mode

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
        scores, ranks, predictions = TransE.predict_samples(self, direct_samples)

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
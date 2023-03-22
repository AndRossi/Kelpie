from typing import Any, Tuple
import numpy
import torch
from torch import nn
from dataset import Dataset
import os
import pandas as pd
from collections import defaultdict

# KEYS FOR SUPPORTED HYPERPARAMETERS (to use in hyperparameter dicts)
DIMENSION = "dimension"                         # embedding dimension, when both entity and relation embeddings have same dimension
ENTITY_DIMENSION = "entity_dimension"           # entity embedding dimension, when entity and relation embeddings have different dimensions
RELATION_DIMENSION = "relation_dimension"       # relation embedding dimension, when entity and relation embeddings have different dimensions

INPUT_DROPOUT = "input_dropout"                 # dropout rate for the input embeddings
HIDDEN_DROPOUT = "hidden_dropout"               # dropout rate after the hidden layer when there is only one hidden layer
HIDDEN_DROPOUT_1 = "hidden_dropout_1"           # dropout rate after the first hidden layer
HIDDEN_DROPOUT_2 = "hidden_dropout_2"           # dropout rate after the second hidden layer
FEATURE_MAP_DROPOUT = "feature_map_dropout"     # feature map dropout

HIDDEN_LAYER_SIZE = "hidden_layer"              # hidden layer size when there is only one hidden layer

INIT_SCALE = "init_scale"                       # downscale to operate on the initial, randomly generated embeddings
OPTIMIZER_NAME = "optimizer_name"               # name of the optimization technique: Adam, Adagrad, SGD
BATCH_SIZE = "batch_size"                       # training batch size
EPOCHS = "epochs"                               # training epochs
LEARNING_RATE = "learning_rate"                 # learning rate
DECAY = "decay"                                 #
DECAY_1 = "decay_1"                             # Adam decay 1
DECAY_2 = "decay_2"                             # Adam decay 2

MARGIN = "margin"                               # pairwise margin-based loss margin
NEGATIVE_SAMPLES_RATIO = "negative_samples"     # number of negative samples to obtain, via corruption, for each positive sample

REGULARIZER_NAME = "regularizer"                # name of the regularization technique: N3
REGULARIZER_WEIGHT = "regularizer_weight"       # weight for the regularization in the loss
LABEL_SMOOTHING = "label_smoothing"             # label smoothing value

GAMMA = "gamma"
RETRAIN_EPOCHS = "retrain_epoches"
kelpie_dataset_cache_size = 30

# global variable!
count_dic = defaultdict(list)
global_dic = {}
relevance_df = pd.DataFrame(columns=['triple', 'explanation', 'relevance', 'head_relevance', 'tail_relevance'])
addition_df = pd.DataFrame(columns=['triple', 'origin', 'addition', 'relevance', 'origin_relevance', 'addition_relevance'])
prelimentary_df = pd.DataFrame(columns=['explanation', 'prelimentary', 'true', 'type_ix'])


def rd(x):
    return round(x, 6)

def path2str(dataset, path):
    if not global_dic['args'].relation_path:
        return ",".join(dataset.sample_to_fact(path))
    s = ''
    for t in path:
        f = dataset.sample_to_fact(t)
        s += f'{f[0]}-{f[1]}->'
    return s + f[2]


def paths2str(dataset, paths):
    return "/".join([path2str(dataset, x) for x in paths])


def strfy(entity_ids):
    if hasattr(entity_ids, '__iter__'):
        lis = [str(x) for x in entity_ids]
        return ','.join(lis)
    return entity_ids

def get_entity_embeddings(entity_embeddings, kelpie_entity_embedding):
    if kelpie_entity_embedding is None:
        return entity_embeddings
    
    # print(len(entity_embeddings), len(kelpie_entity_embedding), entity_embeddings.shape)

    return torch.cat([entity_embeddings, kelpie_entity_embedding], 0)
    # 
    # print(kelpie_entity_embedding.shape, type(kelpie_entity_embedding))
    # return torch.cat(entity_embeddings + [kelpie_entity_embedding], 0)

def terminate_at(length, count):
    '''记录长度为length的解释有多少个'''
    count_dic[length].append(count)
    print(f'\tnumber of rules with length {length}: {count}')


def get_first(x):
    if hasattr(x, "__iter__"):
        return x[0]
    return x

def prefilter_negative(all_rules, top_k=None):
        if type(all_rules) == dict:
            all_rules = all_rules.items()
        all_rules = sorted(all_rules, key=lambda x: get_first(x[1]), reverse=True)
        if top_k is None or top_k > len(all_rules):
            top_k = len(all_rules)
        for i in range(top_k):
            if get_first(all_rules[i][1]) < 0:
                break
        i += 1
        print(f'\tpositive top {top_k} rules: {i}/{len(all_rules)}')
        return all_rules[:i]


def reverse_sample(t: Tuple[Any, Any, Any], num_direct_relations: int):
    if t[1] < num_direct_relations:
        reverse_rel = t[1] + num_direct_relations
    else:
        reverse_rel = t[1] - num_direct_relations
    return (t[2], reverse_rel, t[0])


def get_forward_sample(t: Tuple[Any, Any, Any], num_direct_relations: int):
    if t[1] < num_direct_relations:
        return t
    return (t[2], t[1] - num_direct_relations, t[0])


def plot_dic(dic, path='data/statistic.png', size=(15, 6), label=True, rotation=30, limit=10, hspace=0.4):
    '''
    limit 代表 此值以下不绘制
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(dic), figsize=(size[0], size[1] * len(dic)))
    plt.xticks(rotation=270)
    fig.subplots_adjust(hspace=hspace, wspace=0.2)
    
    for ix, t in enumerate(list(dic.keys())):
        d = {k: v for k, v in dic[t].items() if v >= limit}

        axis = ax if len(dic) == 1 else ax[ix]
        axis.title.set_text(f'{t}: {len(d)}/{len(dic[t])} (limit:{limit})')
        rects = axis.bar([str(x) for x in d.keys()], list(d.values()))
        
        if label:
            for rect in rects:  #rects 是柱子的集合
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), size=10, ha='center', va='bottom')

        for tick in axis.get_xticklabels():
            tick.set_rotation(rotation)
    
    plt.savefig(path)


def plot_dics(dics, folder):
    os.makedirs(folder, exist_ok=True)
    for k, dic in dics.items():
        plot_dic({k: dic}, os.path.join(folder, k + '.png'), limit=0, size=(8, 8), hspace=0, rotation=0)

    

class Model(nn.Module):
    """
        The Model class provides the interface that any LP model should implement.

        The responsibility of any Model implementation is to
            - store the embeddings for entities and relations
            - implement the specific scoring function for that link prediction model
            - offer methods that use that scoring function
                either to run prediction on one or multiple samples, or to run forward propagation in training

        On the contrary, training and evaluation are not performed directly by the model class,
        but require the use of an Optimizer or an Evaluator object respectively.

        All models work with entity and relation ids directly (not with their names),
        that they found from the Dataset object used to initialize the Model.

        Whenever a Model method requires samples, it accepts them in the form of 2-dimensional numpy.arrays,
        where each row corresponds to a sample and contains the integer ids of its head, relation and tail.
    """

    def __init__(self, dataset: Dataset):
        nn.Module.__init__(self)
        self.dataset = dataset

    def is_minimizer(self):
        """
            This method specifies whether this model aims at minimizing of maximizing scores.
            :return: True if in this model low scores are better than high scores; False otherwise.
        """
        pass

    def score(self, samples: numpy.array) -> numpy.array:
        """
            This method computes and returns the plausibility scores for a collection of samples.

            :param samples: a numpy array containing all the samples to score
            :return: the computed scores, as a numpy array
        """
        pass

    # override
    def all_scores(self, samples: numpy.array):
        """
            This method computes, For each of the passed samples, the score for all possible tail entities.
            :param samples: a 2-dimensional numpy array containing the samples to score, one per row
            :return: a 2-dimensional numpy array that, for each sample, contains a row for each passed sample
                     and a column for each possible tail
        """

        pass

    def forward(self, samples: numpy.array):
        """
            This method performs forward propagation for a collection of samples.
            This method is only used in training, when an Optimizer calls it passing the current batch of samples.

            This method returns all the items needed by the Optimizer to perform gradient descent in this training step.
            Such items heavily depend on the specific Model implementation;
            they usually include the scores for the samples (in a form usable by the ML framework, e.g. torch.Tensors)
            but may also include other stuff (e.g. the involved embeddings themselves, that the Optimizer
            may use to compute regularization factors)

            :param samples: a numpy array containing all the samples to perform forward propagation on
        """
        pass

    def predict_samples(self, samples: numpy.array) -> Tuple[Any, Any, Any]:
        """
            This method performs prediction on a collection of samples, and returns the corresponding
            scores, ranks and prediction lists.

            All the passed samples must be DIRECT samples in the original dataset.
            (if the Model supports inverse samples as well,
            it should invert the passed samples while running this method)

            :param samples: the direct samples to predict, in numpy array format
            :return: this method returns three lists:
                        - the list of scores for the passed samples,
                                    OR IF THE MODEL SUPPORTS INVERSE FACTS
                            the list of couples <direct sample score, inverse sample score>,
                            where the i-th score refers to the i-th sample in the input samples.

                        - the list of couples (head rank, tail rank)
                            where the i-th couple refers to the i-th sample in the input samples.

                        - the list of couples (head_predictions, tail_predictions)
                            where the i-th couple refers to the i-th sample in the input samples.
                            The head_predictions and tail_predictions for each sample
                            are numpy arrays containing all the predicted heads and tails respectively for that sample.
        """
        pass

    def predict_sample(self, sample: numpy.array) -> Tuple[Any, Any, Any]:
        """
            This method performs prediction on one (direct) sample, and returns the corresponding
            score, ranks and prediction lists.

            :param sample: the sample to predict, as a numpy array.
            :return: this method returns 3 items:
                    - the sample score
                             OR IF THE MODEL SUPPORTS INVERSE FACTS
                      a couple containing the scores of the sample and of its inverse

                    - a couple containing the head rank and the tail rank

                    - a couple containing the head_predictions and tail_predictions numpy arrays;
                        > head_predictions contains all entities predicted as heads, sorted by decreasing plausibility
                        [NB: the target head will be in this numpy array in position head_rank-1]
                        > tail_predictions contains all entities predicted as tails, sorted by decreasing plausibility
                        [NB: the target tail will be in this numpy array in position tail_rank-1]
        """

        assert sample[1] < self.dataset.num_direct_relations

        scores, ranks, predictions = self.predict_samples(numpy.array([sample]))
        return scores[0], ranks[0], predictions[0]

    def kelpie_model_class(self):
        """
            This method provides the KelpieModel implementation class corresponding to this specific Model class.
            E.g. ComplEx.kelpie_model_class() -> KelpieComplEx.__class__

            When called on a KelpieModel subclass, it raises an exception.

            :return: The KelpieModel class that corresponds to the Model Class running this method
            :raise: an Exception if called on a KelpieModel subclass
        """
        pass


class KelpieModel(Model):
    """
        The KelpieModel class provides the interface that any post-trainable LP model should implement.

        The main functions of KelpieModel are thus identical to Model
        (which is why KelpieModel extends Model).

        In addition to that, a KelpieModels also
    """

    # override
    def predict_samples(self,
                        samples: numpy.array,
                        original_mode: bool = False):
        """
        This method interface overrides the superclass method by adding the option to run predictions in original_mode,
        which means ignoring in any circumstances the additional "fake" kelpie entity.

        :param samples: the DIRECT samples. They will be inverted to perform head prediction
        :param original_mode: a boolean flag specifying whether to work in original_mode or to use the kelpie entity

        :return: a numpy array containing
        """
        pass


    # Override
    def predict_sample(self,
                       sample: numpy.array,
                       original_mode: bool = False):
        """
        This method interface overrides the superclass method by adding the option to run predictions in original_mode,
        which means ignoring in any circumstances the additional "fake" kelpie entity.

        :param sample: the DIRECT sample. It will be inverted to perform head prediction
        :param original_mode: a boolean flag specifying whether to work in original_mode or to use the kelpie entity

        :return:
        """

    # this is necessary
    def update_embeddings(self):
        with torch.no_grad():
            for ix, kelpie_id in enumerate(self.dataset.kelpie_ids):
                self.entity_embeddings[kelpie_id] = self.kelpie_entity_embedding[ix]

    #override
    def train(self, mode=True):
        """
        This method overrides the traditional train() implementation of torch.nn.Module,
        in which a.train() sets all children of a to train mode.

        In KelpieModels, in post-training any layers, including BatchNorm1d or BatchNorm2d,
        must NOT be put in train mode, because even in post-training they MUST remain constant.

        So this method overrides the traditional train() by skipping any layer children
        :param mode:
        :return:
        """
        self.training = mode
        for module in self.children():
            if not (isinstance(module, Model) or
                    isinstance(module, torch.nn.BatchNorm1d) or
                    isinstance(module, torch.nn.BatchNorm2d) or
                    isinstance(module, torch.nn.Linear) or
                    isinstance(module, torch.nn.Conv2d)):
                module.train(mode)
        return self


    def kelpie_model_class(self):
        raise Exception(self.__class__.name  + " is a KelpieModel.")
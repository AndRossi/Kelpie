from typing import Tuple, Any
from dataset import Dataset
from link_prediction.models.model import Model
from prefilters.prefilter import PreFilter

class CriagePreFilter(PreFilter):
    """
    The CriagePreFilter object is a PreFilter that just returns all the samples
    that have as tail the same tail as the sample to explain
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset):
        """
        CriagePreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

        self.tail_2_train_samples = {}

        for (h, r, t) in dataset.train_samples:

            if t not in self.tail_2_train_samples:
                self.tail_2_train_samples[t] = []
            self.tail_2_train_samples[t].append((h, r, t))


    def top_promising_samples_for(self,
                                  sample_to_explain:Tuple[Any, Any, Any],
                                  perspective:str,
                                  top_k=50,
                                  verbose=True):

        """
        This method returns all training samples that have, as a tail,
        either the head or the tail of the sample to explain.

        :param sample_to_explain: the sample to explain
        :param perspective: not used in Criage
        :param top_k: the number of samples to return.
        :param verbose: not used in Criage
        :return: the first top_k extracted samples.
        """

        # note: perspective and verbose will be ignored

        head, relation, tail = sample_to_explain

        tail_as_tail_samples = []
        if tail in self.tail_2_train_samples:
            tail_as_tail_samples = self.tail_2_train_samples[tail]

        head_as_tail_samples = []
        if head in self.tail_2_train_samples:
            head_as_tail_samples = self.tail_2_train_samples[head]

        if top_k == -1:
            return tail_as_tail_samples + head_as_tail_samples
        else:
            return tail_as_tail_samples[:top_k] + head_as_tail_samples[:top_k]

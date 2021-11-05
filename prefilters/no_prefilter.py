from typing import Tuple, Any
from dataset import Dataset
from link_prediction.models.model import Model
from prefilters.prefilter import PreFilter

class NoPreFilter(PreFilter):
    """
    The NoPreFilter object is a fake PreFilter that does not filter away any unpromising facts .
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset):
        """
        NoPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

        self.entity_id_2_train_samples = {}

        for (h, r, t) in dataset.train_samples:

            if h in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[h].append((h, r, t))
            else:
                self.entity_id_2_train_samples[h] = [(h, r, t)]

            if t in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[t].append((h, r, t))
            else:
                self.entity_id_2_train_samples[t] = [(h, r, t)]


    def top_promising_samples_for(self,
                                  sample_to_explain:Tuple[Any, Any, Any],
                                  perspective:str,
                                  top_k = -1,   # not used
                                  verbose=True):

        """
        This method extracts the top k promising samples for interpreting the sample to explain,
        from the perspective of either its head or its tail (that is, either featuring its head or its tail).

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the explanation perspective. It can be either "head" or "tail":
                                - if "head", find the most promising samples featuring the head of the sample to explain
                                - if "tail", find the most promising samples featuring the tail of the sample to explain
        :param top_k: the number of top promising samples to extract.
        :return: the sorted list of the k most promising samples.
        """

        head, relation, tail = sample_to_explain

        if verbose:
            print("Extracting promising facts for" + self.dataset.printable_sample(sample_to_explain))

        start_entity, end_entity = (head, tail) if perspective == "head" else (tail, head)
        samples_featuring_start_entity = self.entity_id_2_train_samples[start_entity]

        return samples_featuring_start_entity

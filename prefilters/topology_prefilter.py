from collections import defaultdict
from typing import Tuple, Any

from dataset import Dataset
from engines.post_training_engine import PostTrainingEngine
from model import Model
from prefilters.prefilter import PreFilter


class PostTrainingPreFilter(PreFilter):
    """
    The PostTrainingPreFilter object is a PreFilter that relies on post-training engine
    to extract the most promising samples for an explanation.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        """
        PostTrainingPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        """
        super().__init__(model, dataset)

        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)

    def top_promising_samples_for(self,
                                  sample_to_explain:Tuple[Any, Any, Any],
                                  perspective:str,
                                  top_k=50):

        """
        This method extracts the top k promising samples for interpreting the sample to explain,
        from the perspective of either its head or its tail (that is, either featuring its head or its tail).

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the explanation perspective. It can be either "head" or "tail":
                                - if "head", find the most promising samples featuring the head of the sample to explain
                                - if "tail", find the most promising samples featuring the tail of the sample to explain
        :param top_k: the number of top promising samples to extract.
        :return: the sorted list of the most promising samples.
        """

        # the post training prefiltering uses relevance in removal to estimate promisingness
        train_samples_with_removal_relevance = self.engine.simple_removal_explanations(sample_to_explain=sample_to_explain,
                                                                                       perspective=perspective)
        print("\tRemoval Relevances: ")
        for x in train_samples_with_removal_relevance:
            print("\t\t" + ";".join(self.dataset.sample_to_fact(x[0])) + ": " + str(x[-1]))
        print()

        top_promising_samples = [x[0] for x in train_samples_with_removal_relevance[:top_k]]
        return top_promising_samples

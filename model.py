from typing import Any, Tuple
import numpy

from dataset import Dataset

class Model:

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

        Different Model implementations may rely on different ML frameworks (e.g. PyTorch, TensorFlow, etc.)

        Whenever a Model method requires samples, it accepts them in the form of 2-dimensional numpy.arrays,
        where each row corresponds to a sample and contains the integer ids of its head, relation and tail.

        NOTE:
            Some models just handle samples as they appear in the original dataset files, while
            some others may also invert them and learn/use inverse versions of the original relations and samples.

            This aspect is part of the model implementation, and should be kept hidden from the other modules.
            Therefore prediction methods (e.g. predict_samples, predict_sample) can only accept direct facts as an input
            (and may ask the dataset to invert them to obtain the head predictions for the original fact).
            This does NOT apply to other methods, such as score and forward.

            The sole exception to this principle is the use of Optimizers.
            Optimizers should have specific implementations for the various models,
            so they are aware of the implementation details of "their" models
            (and can decide to include inverse samples to the model, e.g. this happens all the time in training).
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def score(self, samples: numpy.array) -> numpy.array:
        """
            This method computes and returns the plausibility scores for a collection of samples.

            :param samples: a numpy array containing all the samples to score
            :return: the computed scores, as a numpy array
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
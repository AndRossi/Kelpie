from typing import Tuple, Any
from dataset import Dataset
from prefilters.no_prefilter import NoPreFilter
from prefilters.prefilter import TYPE_PREFILTER, TOPOLOGY_PREFILTER, NO_PREFILTER
from prefilters.type_based_prefilter import TypeBasedPreFilter
from prefilters.topology_prefilter import TopologyPreFilter
from relevance_engines.post_training_engine import PostTrainingEngine
from link_prediction.models.model import Model, global_dic, relevance_df
from explanation_builders.stochastic_necessary_builder import StochasticNecessaryExplanationBuilder
from explanation_builders.stochastic_sufficient_builder import StochasticSufficientExplanationBuilder
import numpy as np

class Kelpie:
    """
    The Kelpie object is the overall manager of the explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations
    to the Pre-Filter, Explanation Builder and Relevance Engine modules.
    """

    DEFAULT_MAX_LENGTH = 4

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 prefilter_type: str,
                 max_explanation_length: int = DEFAULT_MAX_LENGTH):
        """
        Kelpie object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param prefilter_type: the type of prefilter to employ
        :param relevance_threshold: the threshold of relevance that, if exceeded, leads to explanation acceptance
        :param max_explanation_length: the maximum number of facts that the explanations to extract can contain
        """
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.relevance_threshold = global_dic['args'].relevance_threshold
        self.max_explanation_length = max_explanation_length

        print('prefilter type:', prefilter_type)
        if prefilter_type == TOPOLOGY_PREFILTER:
            self.prefilter = TopologyPreFilter(model=model, dataset=dataset)
        elif prefilter_type == TYPE_PREFILTER:
            self.prefilter = TypeBasedPreFilter(model=model, dataset=dataset)
        elif prefilter_type == NO_PREFILTER:
            self.prefilter = NoPreFilter(model=model, dataset=dataset)
        else:
            self.prefilter = TopologyPreFilter(model=model, dataset=dataset)

        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)

    def explain_sufficient(self,
                           sample_to_explain: Tuple[Any, Any, Any],
                           perspective: str,
                           num_promising_samples: int = 50,
                           num_entities_to_convert: int = 10,
                           entities_to_convert: list = None):
        """
        This method extracts sufficient explanations for a specific sample,
        from the perspective of either its head or its tail.

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the perspective of the requested explanations.
                            It can be either "head" or "tail":
                                - if "head", Kelpie answers the question
                                    "given the sample head and relation, why is the sample tail predicted as tail?"
                                - if "tail", Kelpie answers the question
                                    "given the sample relation and tail, why is the sample head predicted as head?"
        :param num_promising_samples: the number of samples relevant to the sample to explain
                                     that must be identified and added to the extracted similar entities
                                     to verify whether they boost the target prediction or not
        :param num_entities_to_convert: the number of entities to convert to extract
                                        (if they have to be extracted)
        :param entities_to_convert: the entities to convert
                                    (if they are passed instead of having to be extracted)

        :return: two lists:
                    the first one contains, for each relevant n-ple extracted, a couple containing
                                - that relevant sample
                                - its value of global relevance across the entities to convert
                    the second one contains the list of entities that the extractor has tried to convert
                        in the sufficient explanation process

        """

        most_promising_samples = self.prefilter.top_promising_samples_for(sample_to_explain=sample_to_explain,
                                                                          perspective=perspective,
                                                                          top_k=num_promising_samples)

        explanation_builder = StochasticSufficientExplanationBuilder(model=self.model,
                                                                     dataset=self.dataset,
                                                                     hyperparameters=self.hyperparameters,
                                                                     sample_to_explain=sample_to_explain,
                                                                     perspective=perspective,
                                                                     num_entities_to_convert=num_entities_to_convert,
                                                                     entities_to_convert=entities_to_convert,
                                                                     relevance_threshold=self.relevance_threshold,
                                                                     max_explanation_length=self.max_explanation_length)

        explanations_with_relevance = explanation_builder.build_explanations(samples_to_add=most_promising_samples)
        return explanations_with_relevance, explanation_builder.entities_to_convert

    def explain_necessary(self,
                          sample_to_explain: Tuple[Any, Any, Any],
                          perspective: str,
                          num_promising_samples: int = 0):
        """
        This method extracts necessary explanations for a specific sample,
        from the perspective of either its head or its tail.

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the perspective of the requested explanations.
                            It can be either "head" or "tail":
                                - if "head", Kelpie answers the question
                                    "given the sample head and relation, why is the sample tail predicted as tail?"
                                - if "tail", Kelpie answers the question
                                    "given the sample relation and tail, why is the sample head predicted as head?"
        :param num_promising_samples: the number of samples relevant to the sample to explain
                                     that must be identified and removed from the entity under analysis
                                     to verify whether they worsen the target prediction or not

        :return: a list containing for each relevant n-ple extracted, a couple containing
                                - that relevant n-ple
                                - its value of relevance

        """

        most_promising_samples = self.prefilter.top_promising_samples_for(sample_to_explain=sample_to_explain,
                                                                          perspective=perspective,
                                                                          top_k=num_promising_samples)

        explanation_builder = StochasticNecessaryExplanationBuilder(model=self.model,
                                                                    dataset=self.dataset,
                                                                    hyperparameters=self.hyperparameters,
                                                                    sample_to_explain=sample_to_explain,
                                                                    perspective=perspective,
                                                                    relevance_threshold=self.relevance_threshold,
                                                                    max_explanation_length=self.max_explanation_length)
        
        explanations_with_relevance = explanation_builder.build_explanations(samples_to_remove=most_promising_samples)
        
        for rule, relevance in explanations_with_relevance:
            relevance_df.loc[len(relevance_df)] = {
                'triple': sample_to_explain,
                'explanation': rule,
                'relevance': relevance[0],
                'head_relevance': relevance[1],
                'tail_relevance': relevance[2],
            }

        return explanations_with_relevance

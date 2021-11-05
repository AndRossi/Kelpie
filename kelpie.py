from typing import Tuple, Any
from dataset import Dataset
from relevance_engines.post_training_engine import PostTrainingEngine
from link_prediction.models.model import Model
from prefilters.topology_prefilter import TopologyPreFilter
from explanation_builders.stochastic_necessary_builder import StochasticNecessaryExplanationBuilder
from explanation_builders.stochastic_sufficient_builder import StochasticSufficientExplanationBuilder

class Kelpie:
    """
    The Kelpie object is the overall manager of the explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations
    to the Pre-Filter, Explanation Builder and Relevance Engine modules.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        """
        Kelpie object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        """
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters

        self.prefilter = TopologyPreFilter(model=model,
                                           dataset=dataset)
                                           #hyperparameters=hyperparameters)
        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)

    def explain_sufficient(self,
                           sample_to_explain:Tuple[Any, Any, Any],
                           perspective:str,
                           num_promising_samples=50,
                           num_entities_to_convert=10,
                           entities_to_convert = None):
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
                                                                entities_to_convert = entities_to_convert)

        explanations_with_relevance = explanation_builder.build_explanations(samples_to_add=most_promising_samples)
        return explanations_with_relevance, explanation_builder.entities_to_convert

    def explain_necessary(self,
                          sample_to_explain:Tuple[Any, Any, Any],
                          perspective:str,
                          num_promising_samples=50):
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
                                                               perspective=perspective)

        explanations_with_relevance = explanation_builder.build_explanations(samples_to_remove=most_promising_samples)
        return explanations_with_relevance

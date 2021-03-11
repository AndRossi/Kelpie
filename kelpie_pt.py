from typing import Tuple, Any
from dataset import Dataset
from engines.post_training_engine import PostTrainingEngine
from model import Model
#from prefilters.pt_prefilter import PostTrainingPreFilter
from prefilters.topology_prefilter import TopologyPreFilter
from rule_extraction.probabilistic_extractor import ProbabilisticSufficientRuleExtractor


class Kelpie:
    """
    The Kelpie object is the overall manager of the explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations to the ExplanationEngines
    and to the entity_similarity modules.
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

    def explain(self,
                sample_to_explain:Tuple[Any, Any, Any],
                perspective:str,
                num_promising_samples=50,
                num_entities_to_convert=10):
        """
        This method extracts explanations for a specific sample, from the perspective of either its head or its taiL,

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

        :param num_entities_to_convert: the number of entities that must be extracted and converted by the sufficient rules
        :return: two lists:
                    the first one contains, for each relevant sample, a couple containing the sample and an index of its global relevance across the similar entities
                    the second one contains, for each combination of relevant samples, a couple containing the combination and its relevance across the similar entities

        """

        most_promising_samples = self.prefilter.top_promising_samples_for(sample_to_explain=sample_to_explain,
                                                                          perspective=perspective,
                                                                          top_k=num_promising_samples)

        #rule_extractor = BruteForceSufficientRuleExtractor(model=self.model,

        rule_extractor = ProbabilisticSufficientRuleExtractor(model=self.model,
                                                           dataset=self.dataset,
                                                           hyperparameters=self.hyperparameters,
                                                           sample_to_explain=sample_to_explain,
                                                           perspective=perspective,
                                                           num_entities_to_convert=num_entities_to_convert)

        rules_with_relevance = rule_extractor.extract_rules(samples_to_add=most_promising_samples)
        return rules_with_relevance, rule_extractor.entities_to_convert

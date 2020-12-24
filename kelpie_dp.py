from collections import defaultdict
from typing import Tuple, Any

from dataset import Dataset
from engines.entity_similarity import extract_comparable_entities
from engines.data_poisoning_engine import DataPoisoningEngine
from model import Model, LEARNING_RATE


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
        self.engine = DataPoisoningEngine(model=model,
                                     dataset=dataset,
                                     hyperparameters=hyperparameters,
                                     epsilon=hyperparameters[LEARNING_RATE])

    def explain(self,
                sample_to_explain:Tuple[Any, Any, Any],
                perspective:str,
                num_similar_entities=10,
                num_relevant_samples=50):
        """
        This method extracts explanations for a specific sample, from the perspective of either its head or its taiL,

        The explanation pipeline is the following:
            1. extract the top relevant training samples for the sample_to_explain
            2. identify a set of specimen entities, based on their similarity to
               either the sample head or the sample tail (depending on the chosen perspective),
               and for which the model currently does not predict the same prediction as the sample to explain.
            3. identify which of the training samples extracted in phase 1, if added to the entities extracted in phase 2,
               can boost their prediction making them more similar to the prediction of the sample to explain.
            4. identify which combinations of the training samples extracted in phase 1,
               if added to the entities extracted in phase 2,
               can boost their prediction making them more similar to the prediction of the sample to explain.

        For instance, if the sample to explain is <Barack Obama, nationality, USA> and the perspective is "head",
        Kelpie finds explanations to the question "Given Barack Obama and nationality, why does the model predict USA?":
            1. it extracts the training samples featuring Barack Obama
               that are most relevant to <Barack Obama, nationality, USA>;
            2. it extracts a set of specimen entities "e" based on their similarity to Barack Obama,
               conditioned to relation nationality and tail USA,
               and for which <"e", nationality, ?> does not rank "USA" as tail in first position (filtered scenario)
            3. it identifies which of the Barack Obama samples extracted in phase 1,
               if added to the entities "e" extracted in phase 2, can boost the <"e", nationality, USA> prediction
            4. identify which combinations of the of the Barack Obama samples extracted in 1,
               if added to the "e" entities extracted in phase 2, can boost the <"e", nationality, USA> prediction.

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the perspective of the requested explanations.
                            It can be either "head" or "tail":
                                - if "head", Kelpie answers the question
                                    "given the sample head and relation, why is the sample tail predicted as tail?"
                                - if "tail", Kelpie answers the question
                                    "given the sample relation and tail, why is the sample head predicted as head?"
        :param num_similar_entities: the number of entities that must be extracted
                                     based on their similarity to the perspective entity
        :param num_relevant_samples: the number of samples relevant to the sample to explain
                                     that must be identified and added to the extracted similar entities
                                     to verify whether they boost the target prediction or not
        :return: two lists:
                    the first one contains, for each relevant sample, a couple containing the sample and an index of its global relevance across the similar entities
                    the second one contains, for each combination of relevant samples, a couple containing the combination and its relevance across the similar entities

        """

        # identify the perspective entity entity in the sample to explain
        head, relation, tail = sample_to_explain
        perspective_entity = head if perspective == "head" else tail

        # extract the training samples featuring the perspective entity and most relevant to the sample to explain
        train_samples_with_relevance = self.engine.simple_removal_explanations(sample_to_explain=sample_to_explain,
                                                                               perspective=perspective)
        most_relevant_train_samples = [x[0] for x in train_samples_with_relevance[:num_relevant_samples]]

        # extract the top entities most "comparable" to the perspective entity
        entities_with_comparability = extract_comparable_entities(model=self.model,
                                                                  dataset=self.dataset,
                                                                  sample=sample_to_explain,
                                                                  perspective=perspective,
                                                                  num_entities=num_similar_entities,
                                                                  policy="best")
        comparable_entities = [x[0] for x in entities_with_comparability]

        original_sample_2_coverage = defaultdict(lambda: 0.0)
        original_couple_2_coverage = defaultdict(lambda: 0.0)

        # for each comparable entity
        # add each of the most_relevant_train_samples to the comparable entity
        # and individually check its relevance
        for comparable_entity in comparable_entities:

            # replace the perspective entity with the comparable_entity
            # both in the sample to explain and in all the samples to add
            comparable_sample_to_explain = Dataset.replace_entity_in_sample(sample_to_explain, perspective_entity, comparable_entity)

            # replace the perspective entity with the comparable_entity
            comparable_samples_to_add = Dataset.replace_entity_in_samples(samples=most_relevant_train_samples,
                                                                          old_entity=perspective_entity,
                                                                          new_entity=comparable_entity,
                                                                          as_numpy=False)

            comparable_samples_with_relevance = self.engine.simple_addition_explanations(sample_to_explain=comparable_sample_to_explain,
                                                                                         perspective=perspective,
                                                                                         samples_to_add=comparable_samples_to_add)

            for i, cur_comparable_sample_with_relevance in enumerate(comparable_samples_with_relevance):
                # for each cur_comparable_sample
                cur_comparable_sample = cur_comparable_sample_with_relevance[0]

                # go back to the sample containing the perspective entity
                cur_sample = Dataset.replace_entity_in_sample(sample=cur_comparable_sample,
                                                              old_entity=comparable_entity,
                                                              new_entity=perspective_entity,
                                                              as_numpy=False)
                # update the coverage value of the cur_sample
                original_sample_2_coverage[cur_sample] += 1.0/(float(i+1.0)*len(comparable_entities))


        expl_samples = sorted(original_sample_2_coverage.items(), key=lambda x:x[1], reverse=True)[:10]
        expl_couples = sorted(original_couple_2_coverage.items(), key=lambda x:x[1], reverse=True)[:10]

        return expl_samples, expl_couples, comparable_entities
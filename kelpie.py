from collections import defaultdict
from typing import Tuple, Any
from dataset import Dataset
from engines.entity_similarity import extract_comparable_entities
from engines.gradient_engine import GradientEngine
from model import Model, LEARNING_RATE


class Kelpie:

    XSI_THRESHOLD = 0.7
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
        self.engine = GradientEngine(model=model,
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

        # extract the training samples featuring the perspective entity and most relevant to the sample to explain
        train_samples_with_relevance = self.engine.simple_removal_explanations(sample_to_explain=sample_to_explain,
                                                                               perspective=perspective,
                                                                               top_k=num_relevant_samples)
        most_relevant_train_samples = [x[0] for x in train_samples_with_relevance]

        # extract the top entities most "comparable" to the perspective entity
        entities_with_comparability = extract_comparable_entities(model=self.model,
                                                                  dataset=self.dataset,
                                                                  sample=sample_to_explain,
                                                                  perspective=perspective,
                                                                  num_entities=num_similar_entities,
                                                                  policy="random")
        comparable_entities = [x[0] for x in entities_with_comparability]


        terminate = False
        rule_length = 1

        while not terminate and rule_length < 5:

            ### global maps

            # map each original added sample to the coverage obtained
            # (computed based on the rank of that added sample across all comparable entities)
            original_added_combination_2_coverage = defaultdict(lambda: 0.0)

            # map each comparable entity to a nested map
            # that associates each original added sample to the corresponding xsi value
            comparable_entity_2_xsi_metric_map = defaultdict(lambda: dict())

            # for each comparable entity, add each of the most_relevant_train_samples to the comparable entity
            # and individually check its relevance
            for comparable_entity in comparable_entities:

                original_added_combinations_with_relevance, \
                original_added_combination_2_perturbed_score, \
                target_entity_score, \
                best_entity_score = self.extract_combinations_to_convert_entity(original_sample_to_explain=sample_to_explain,
                                                                                original_samples_to_add=most_relevant_train_samples,
                                                                                perspective=perspective,
                                                                                entity_to_convert=comparable_entity,
                                                                                combination_length=rule_length)
                # for each added sample:
                #   1) update its coverage value in the global map
                #   2) update its xsi value for the current comparable entity the global map
                for i, original_added_sample_with_relevance in enumerate(original_added_combinations_with_relevance):
                    original_added_sample = original_added_sample_with_relevance[0]
                    original_added_combination_2_coverage[original_added_sample] += 1.0/(float(i+1.0)*len(comparable_entities))

                    perturbed_score = original_added_combination_2_perturbed_score[original_added_sample]
                    original_added_sample_xsi = self.compute_xsi(target_entity_score, perturbed_score, best_entity_score)
                    comparable_entity_2_xsi_metric_map[comparable_entity][original_added_sample] = original_added_sample_xsi

            best_added_combination, coverage = sorted(original_added_combination_2_coverage.items(), key=lambda x:x[1], reverse=True)[0]
            xsis = [comparable_entity_2_xsi_metric_map[x][best_added_combination] for x in comparable_entities]

            termination = True
            for xsi in xsis:
                if xsi < self.XSI_THRESHOLD:
                    termination = False
                    break

            if termination:
                return best_added_combination, comparable_entities

        return best_added_combination, comparable_entities


    def explain_old(self,
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

        outlines_simple = []
        outlines_comb = []

        # identify the perspective entity entity in the sample to explain
        head, relation, tail = sample_to_explain
        perspective_entity = head if perspective == "head" else tail

        # extract the training samples featuring the perspective entity and most relevant to the sample to explain
        train_samples_with_relevance = self.engine.simple_removal_explanations(sample_to_explain=sample_to_explain,
                                                                               perspective=perspective,
                                                                               top_k=num_relevant_samples)

        most_relevant_train_samples = [x[0] for x in train_samples_with_relevance[:num_relevant_samples]]

        # extract the top entities most "comparable" to the perspective entity
        entities_with_comparability = extract_comparable_entities(model=self.model,
                                                                  dataset=self.dataset,
                                                                  sample=sample_to_explain,
                                                                  perspective=perspective,
                                                                  num_entities=num_similar_entities,
                                                                  policy="random")
        comparable_entities = [x[0] for x in entities_with_comparability]

        original_sample_2_coverage = defaultdict(lambda: 0.0)
        original_nple_2_coverage = defaultdict(lambda: 0.0)

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


            # obtain the relevance of those samples in addition, with respect to the the sample to explain
            comparable_samples_with_relevance, \
            comparable_sample_2_perturbed_score, \
            target_entity_score, best_entity_score  = self.engine.simple_addition_explanations(sample_to_explain=comparable_sample_to_explain,
                                                                                               perspective=perspective,
                                                                                               samples_to_add=comparable_samples_to_add)

            for comparable_sample in comparable_sample_2_perturbed_score:
                comparable_triple = self.dataset.sample_to_fact(comparable_sample)
                outlines_simple.append(";".join(comparable_triple) + ";" + \
                                        str(best_entity_score) + ";" + \
                                        str(target_entity_score) + ";" + \
                                        str(comparable_sample_2_perturbed_score[comparable_sample]) + "\n")


            for i, cur_comparable_sample_with_relevance in enumerate(comparable_samples_with_relevance):
                # for each cur_comparable_sample

                cur_comparable_sample, cur_relevance = cur_comparable_sample_with_relevance
                # go back to the sample containing the perspective entity
                cur_sample = Dataset.replace_entity_in_sample(sample=cur_comparable_sample,
                                                              old_entity=comparable_entity,
                                                              new_entity=perspective_entity,
                                                              as_numpy=False)

                original_sample_2_coverage[cur_sample] += 1.0/(float(i+1.0)*len(comparable_entities))


            out = self.engine.nple_addition_explanations(sample_to_explain=comparable_sample_to_explain,
                                                                                             perspective=perspective,
                                                                                             samples_to_add=comparable_samples_to_add,
                                                                                             n=2)
            if len(out) == 4:
                comparable_nples_with_relevance, \
                comparable_nple_2_perturbed_score, \
                target_entity_score, best_entity_score = out
            else:
                continue

            for comparable_nple in comparable_nple_2_perturbed_score:
                outlines_comb.append("+".join([";".join(self.dataset.sample_to_fact(x)) for x in comparable_nple]) + ";" + \
                                     str(best_entity_score) + ";" + \
                                     str(target_entity_score) + ";" + \
                                     str(comparable_nple_2_perturbed_score[comparable_nple]) + "\n")

            for i, cur_comparable_nple_with_relevance in enumerate(comparable_nples_with_relevance):
                cur_comparable_nple = cur_comparable_nple_with_relevance[0]
                cur_nple = tuple([Dataset.replace_entity_in_sample(sample=cur_sample,
                                                                   old_entity=comparable_entity,
                                                                   new_entity=perspective_entity,
                                                                   as_numpy=False) for cur_sample in cur_comparable_nple])
                original_nple_2_coverage[cur_nple] += 1.0/(float(i+1.0)*len(comparable_entities))

        expl_samples = sorted(original_sample_2_coverage.items(), key=lambda x:x[1], reverse=True)[:10]
        expl_nples = sorted(original_nple_2_coverage.items(), key=lambda x:x[1], reverse=True)[:10]

        with open("output_details_1.csv", "a") as output_simple:
            output_simple.writelines(outlines_simple)

        with open("output_details_comb.csv", "a") as output_comb:
            output_comb.writelines(outlines_comb)

        return expl_samples, expl_nples, comparable_entities



    def extract_combinations_to_convert_entity(self,
                                               original_sample_to_explain,
                                               original_samples_to_add,
                                               perspective: str,
                                               entity_to_convert,
                                               combination_length):

        original_entity = original_sample_to_explain[0] if perspective == "head" else original_sample_to_explain[2]

        # replace the perspective entity with the comparable_entity
        # both in the sample to explain and in all the samples to add
        # and keep trace of the mappings in two support data structures

        # support data structures
        original_sample_2_repl_sample = dict()
        repl_sample_2_original_sample = dict()

        # replace the entity to explain with the entity to convert in the original sample to explain
        sample_to_convert = Dataset.replace_entity_in_sample(sample=original_sample_to_explain,
                                                             old_entity=original_entity,
                                                             new_entity=entity_to_convert)

        # replace the entity to explain with the entity to convert in the original samples to add
        repl_samples_to_add = Dataset.replace_entity_in_samples(samples=original_samples_to_add,
                                                                old_entity=original_entity,
                                                                new_entity=entity_to_convert,
                                                                as_numpy=False)

        # populate the support data structures
        for i in range(len(original_samples_to_add)):
            original_sample_2_repl_sample[original_samples_to_add[i]] = repl_samples_to_add[i]
            repl_sample_2_original_sample[repl_samples_to_add[i]] = original_samples_to_add[i]


        if combination_length == 1:
            # obtain the relevance of those samples in addition, with respect to the the sample to explain
            repl_added_samples_with_relevance, \
            repl_added_sample_2_perturbed_score, \
            target_entity_score, \
            best_entity_score = self.engine.simple_addition_explanations(sample_to_explain=sample_to_convert,
                                                                         perspective=perspective,
                                                                         samples_to_add=repl_samples_to_add)

            original_added_samples_with_relevance = [(Dataset.replace_entity_in_sample(a, entity_to_convert, original_entity, False), b) for (a, b) in repl_added_samples_with_relevance]
            original_added_sample_2_perturbed_score = {Dataset.replace_entity_in_sample(a, entity_to_convert, original_entity, False):repl_added_sample_2_perturbed_score[a] for a in repl_added_sample_2_perturbed_score}
            return original_added_samples_with_relevance, original_added_sample_2_perturbed_score, target_entity_score, best_entity_score

        else:
            repl_added_nples_with_relevance, \
            repl_added_nple_2_perturbed_score, \
            target_entity_score, \
            best_entity_score = self.engine.nple_addition_explanations(sample_to_explain=sample_to_convert,
                                                                       perspective=perspective,
                                                                       samples_to_add=repl_samples_to_add,
                                                                       n=combination_length)

            original_added_samples_with_relevance = [(Dataset.replace_entity_in_sample(a, entity_to_convert, original_entity, False), b) for (a, b) in repl_added_nples_with_relevance]
            original_added_nple_2_perturbed_score = {Dataset.replace_entity_in_sample(a, entity_to_convert, original_entity, False):repl_added_nple_2_perturbed_score[a] for a in repl_added_nple_2_perturbed_score}
            return original_added_samples_with_relevance, original_added_nple_2_perturbed_score, target_entity_score, best_entity_score



    def compute_xsi(self,
                    original_target_score: float,
                    perturbed_target_score: float,
                    original_best_score: float):

        term_1 = (perturbed_target_score - original_target_score)/perturbed_target_score
        term_2 = (original_best_score - original_target_score)/original_best_score

        xsi = term_1 + term_2
        return xsi
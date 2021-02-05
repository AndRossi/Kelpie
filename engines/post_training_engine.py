import math
from typing import Type, Tuple, Any

import numpy

from dataset import KelpieDataset, Dataset
from engines.engine import ExplanationEngine
from model import Model, KelpieModel

class PostTrainingEngine(ExplanationEngine):

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 post_training_optimizer_class: Type,
                 hyperparameters: dict):
        """
            PostTrainingEngine constructor.

            :param model: the trained Model to explain the behaviour of. This can NOT be a KelpieModel.
            :param dataset: the Dataset used to train the model
            :param hyperparameters: dict containing all the hyperparameters necessary for running the post-training
                                    (for both the model and the optimizer)
        """

        ExplanationEngine.__init__(self, model=model, dataset=dataset, hyperparameters=hyperparameters)
        if isinstance(model, KelpieModel):
            raise Exception("The model passed to the PostTrainingEngine is already a post-trainable KelpieModel.")

        self.kelpie_optimizer_class = post_training_optimizer_class

    def simple_removal_explanations(self,
                                    sample_to_explain: Tuple[Any, Any, Any],
                                    perspective: str,
                                    top_k: int):
        """
            Given a sample to explain, and the perspective from which to explain it,
            find the k training samples containing the perspective entity that, if removed (one by one)
            would affect the most the prediction of the sample to explain.

            :param sample_to_explain: the sample to explain in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param top_k: the number of top relevant training samples to return

            :return: an array of k pairs, where each pair is a relevant training sample with its relevance value,
                     sorted by descending relevance
        """

        head_id, relation_id, tail_id = sample_to_explain
        original_entity_id = head_id if perspective == "head" else tail_id

        # extract all training samples containing the entity to explain
        if self.dataset.entity_2_degree[original_entity_id] == 0:
            return None

        # create a Kelpie Dataset focused on the original entity to explain
        kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_id=original_entity_id)
        kelpie_sample_to_explain = kelpie_dataset.as_kelpie_sample(original_sample=sample_to_explain)

        # Create a "clone" of the entity to explain, post-train it and extract the direct score of the clone
        print("\tEntity to explain: " + self.dataset.entity_id_2_name[original_entity_id] + " (degree: " + str(self.dataset.entity_2_degree[original_entity_id]) + ")")
        print("\tRunning base post-training of that entity...")
        base_pt_model = self.post_train(kelpie_dataset=kelpie_dataset, kelpie_train_samples=kelpie_dataset.kelpie_train_samples) # type: KelpieModel

        # check how the post-trained "clone" performs on the sample to explain
        base_pt_model.eval()    # important!
        (base_direct_score, base_inverse_score), \
        (base_head_rank, base_tail_rank), _ = base_pt_model.predict_sample(sample=kelpie_sample_to_explain, original_mode=False)

        results = []
        # skip the kelpie_train_samples one by one and perform post-training
        for i in range(len(kelpie_dataset.kelpie_train_samples)):
            current_kelpie_training_samples = numpy.vstack((kelpie_dataset.kelpie_train_samples[:i], kelpie_dataset.kelpie_train_samples[i+1:]))

            # save the skipped training sample in its non-kelpie version (that is, using the original entity id)
            skipped_training_sample = kelpie_dataset.as_original_sample(kelpie_sample=kelpie_dataset.kelpie_train_samples[i])

            post_trained_kelpie_model = self.post_train(kelpie_dataset=kelpie_dataset, kelpie_train_samples=current_kelpie_training_samples)

            post_trained_kelpie_model.eval() # important!
            (cur_direct_score, cur_inverse_score), \
            (cur_head_rank, cur_tail_rank), _ = post_trained_kelpie_model.predict_sample(sample=kelpie_sample_to_explain, original_mode=False)
            print("\tIteration " + str(i) + ": skipping sample <"+ ", ".join(kelpie_dataset.sample_to_fact(skipped_training_sample)) + ">")

            # we want to give higher priority to the facts that, when removed, worsen the score the most. So:
            # if the model is a minimizer the greater cur_direct_score is than base_direct_score, the more relevant the removed fact
            if self.model.is_minimizer():
                score_worsening = cur_direct_score - base_direct_score
            # if the model is a maximizer that lesser cur_direct_score is than base_direct_score, the more relevant the removed fact
            else:
                score_worsening = base_direct_score - cur_direct_score

            #print("\t\tDirect score in the post-trained model: " + str(cur_direct_score) + "(base was" + str(base_direct_score) + ")")
            #print("\t\tDirect tail score worsening: " + str(score_worsening))

            rank_worsening = cur_tail_rank - base_tail_rank
            #print("\t\tRank worsening: " + str(rank_worsening))

            relevance = rank_worsening + self.sigmoid(score_worsening)
            #print("\t\tRelevance: " + str(relevance))

            results.append((skipped_training_sample, cur_direct_score, cur_inverse_score, cur_head_rank, cur_tail_rank, relevance))

        results = sorted(results, key=lambda element: element[-1], reverse=True)

        if top_k == -1 or top_k < len(results):
            return results
        else:
            return results[:top_k]


    def simple_addition_explanations(self,
                                     sample_to_convert: Tuple[Any, Any, Any],
                                     perspective: str,
                                     samples_to_add: list):
        """
            Given a "sample to convert" (that is, a sample that the model currently does not predict as true,
            and that we want to be predicted as true); given the perspective from which to intervene on it;
            and given and a list of other training samples containing the perspective entity;
            for each sample in the list, compute an esteem of the relevance it would have if added to the perspective entity
            to improve the most the prediction of the sample to convert.

            :param sample_to_convert: the sample that we would like the model to predict as "true",
                                      in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param samples_to_add: the list of samples containing the perspective entity
                                   that we want to analyze the effect of, if added to the perspective entity

            :return: an array of pairs, where each pair is a samples_to_add with the extracted value of relevance,
                     sorted by descending relevance
        """

        head_id, relation_id, tail_id = sample_to_convert
        original_entity_id = head_id if perspective == "head" else tail_id

        # check how the original model performs on the original sample to explain
        original_target_entity_score, \
        original_best_entity_score, \
        original_target_entity_rank = self.extract_detailed_performances_on_sample(self.model, sample_to_convert)

        # create a Kelpie Dataset focused on the original id of the entity to explain
        kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_id=original_entity_id)

        # the kelpie sample to convert features the kelpie entity rather than the "real" one
        kelpie_sample_to_convert = kelpie_dataset.as_kelpie_sample(original_sample=sample_to_convert)

        print("\t\tRunning base post-training before additions...")
        base_pt_model = self.post_train(kelpie_dataset=kelpie_dataset, kelpie_train_samples=kelpie_dataset.kelpie_train_samples) # type: KelpieModel

        # then check how the base post-trained model performs on the kelpie sample to explain.
        # This means checking how the "clone entity" (with no additional samples) performs
        base_pt_target_entity_score, \
        base_pt_best_entity_score, \
        base_pt_target_entity_rank = self.extract_detailed_performances_on_sample(base_pt_model, kelpie_sample_to_convert)

        # finally, check how the kelpie post-trained models perform on the kelpie sample to explain.
        # This means checking how the "kelpie entities", each with its specific addition, perform.
        added_sample_2_relevance = dict()
        added_sample_2_pt_results = dict()
        for i in range(len(samples_to_add)):
            cur_sample_to_add = samples_to_add[i]
            print("\t\tIteration " + str(i) + ": adding sample <"+ ", ".join(self.dataset.sample_to_fact(cur_sample_to_add)) + ">")

            # we should now create a new Kelpie Dataset identical to kelpie_dataset but also containing cur_sample_to_add.
            # However this would be awfully time-consuming.
            # For the sake of efficiency, we will just temporarily add those facts to the current kelpie_dataset
            # and then undo the addition after the post-training and the evaluation of the post-trained model are over.
            kelpie_dataset.add_training_samples(numpy.array([cur_sample_to_add]))   # no need to add the "kelpie" version of cur_sample_to_add: the "add_training_samples" method replaces the original entity with the kelpie entity by itself

            # post-train a kelpie model on the dataset that has undergone the addition
            cur_kelpie_model = self.post_train(kelpie_dataset=kelpie_dataset, kelpie_train_samples=kelpie_dataset.kelpie_train_samples)  # type: KelpieModel

            # then check how the post-trained model performs on the kelpie sample to explain.
            # This means checking how the "kelpie entity" (with the added sample) performs, rather than the original entity
            pt_target_entity_score, \
            pt_best_entity_score, \
            pt_target_entity_rank = self.extract_detailed_performances_on_sample(cur_kelpie_model, kelpie_sample_to_convert)

            # undo the addition, to allow the following iterations of this loop
            kelpie_dataset.undo_last_training_samples_addition()

            # we want to give higher priority to the facts that, when added, make the score the better. So:
            # if the model is a minimizer the smaller pt_target_entity_score is than base_pt_target_entity_score, the more relevant the added fact
            if self.model.is_minimizer():
                score_worsening = base_pt_target_entity_score - pt_target_entity_score
            # if the model is a maximizer, the greater pt_target_entity_score is than base_pt_target_entity_score, the more relevant the added fact
            else:
                score_worsening = pt_target_entity_score - base_pt_target_entity_score

            relevance = base_pt_target_entity_rank - pt_target_entity_rank + self.sigmoid(score_worsening)

            added_sample_2_relevance[cur_sample_to_add] = relevance
            added_sample_2_pt_results[cur_sample_to_add] = (pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank)

        sorted_samples_with_relevance = sorted(added_sample_2_relevance.items(), key=lambda element: element[-1], reverse=True)

        return sorted_samples_with_relevance, \
               added_sample_2_pt_results, \
               original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
               base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank

    def nple_addition_explanations(self,
                                   sample_to_convert: Tuple[Any, Any, Any],
                                   perspective: str,
                                   samples_to_add: list,
                                   n: int):
        """
            Given a "sample to convert" (that is, a sample that the model currently does not predict as true,
            and that we want to be predicted as true); given the perspective from which to intervene on it;
            given and a list of other training samples containing the perspective entity;
            and given the length "n" of combinations of training samples to try;
            for each combination of length "n" of the samples in the list,
            estimate the relevance that adding that combination of facts to the perspective entity
            would have to improve the prediction of the sample to convert.

            :param sample_to_convert: the sample that we would like the model to predict as "true",
                                      in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param samples_to_add: the list of samples containing the perspective entity
                                   that we want to analyze the effect of, if added to the perspective entity
            :param n: length of combinations of samples to add to the perspective entity

            :return: an array of pairs, where each pair is a n n-long combination of samples_to_add
                     coupled with with the corresponding extracted value of relevance, sorted by descending relevance
        """

        # if there are less than n items in samples_to_add
        # it is not possible to try combinations of length n of samples_to_add
        if len(samples_to_add) < n:
            return []

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_convert
        original_entity_id = head_id if perspective == "head" else tail_id

        # check how the original model performs on the original sample to explain
        original_target_entity_score, \
        original_best_entity_score, \
        original_target_entity_rank = self.extract_detailed_performances_on_sample(self.model, sample_to_convert)

        # create a Kelpie Dataset focused on the original id of the entity to explain
        kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_id=original_entity_id)
        # the kelpie sample to convert features the kelpie entity rather than the "real" one
        kelpie_sample_to_convert = kelpie_dataset.as_kelpie_sample(original_sample=sample_to_convert)

        print("### BASE POST-TRAINING")
        base_pt_model = self.post_train(kelpie_dataset=kelpie_dataset, kelpie_train_samples=kelpie_dataset.kelpie_train_samples) # type: KelpieModel

        # then check how the base post-trained model performs on the kelpie sample to explain.
        # This means checking how the "clone entity" (with no additional samples) performs
        base_pt_target_entity_score, \
        base_pt_best_entity_score, \
        base_pt_target_entity_rank = self.extract_detailed_performances_on_sample(base_pt_model, kelpie_sample_to_convert)

        # extract all n-long different combinations of samples
        samples_nples = self._extract_sample_nples(samples=samples_to_add, n=n)

        added_nple_2_perturbed_score = {}
        added_nple_2_relevance = {}

        for i in range(len(samples_nples)):
            current_nple_to_add = samples_nples[i]

            # we should now create a new Kelpie Dataset identical to the kelpie_dataset, but also containing the current_nple_to_add.
            # However this would be awfully time-consuming for a dataset that will only be used for one post-training and evaluation.
            # So, for the sake of efficiency, we will just temporarily add those facts to the current kelpie_dataset
            # and then undo the addition as soon as we don't need them anymore.
            cur_nple_to_add_numpy_arr = numpy.array(current_nple_to_add)
            kelpie_dataset.add_training_samples(cur_nple_to_add_numpy_arr)

            cur_kelpie_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                               kelpie_train_samples=kelpie_dataset.kelpie_train_samples)  # type: KelpieModel

            # then check how the post-trained model performs on the kelpie sample to explain.
            # This means checking how the "kelpie entity" (with the added sample) performs, rather than the original entity
            pt_target_entity_score, \
            pt_best_entity_score, \
            pt_target_entity_rank = self.extract_detailed_performances_on_sample(cur_kelpie_model, kelpie_sample_to_convert)

            # undo the addition, to allow the following iterations of this loop
            kelpie_dataset.undo_last_training_samples_addition()

            if self.model.is_minimizer():
                direct_score_improvement = base_pt_target_entity_score - pt_target_entity_score
            else:
                direct_score_improvement = pt_target_entity_score - base_pt_target_entity_score

            added_nple_2_relevance[current_nple_to_add] = direct_score_improvement
            added_nple_2_perturbed_score[current_nple_to_add] = pt_target_entity_score

        sorted_nples_with_relevance = sorted(added_nple_2_relevance.items(), key=lambda element: element[-1], reverse=True)

        return sorted_nples_with_relevance, \
               added_nple_2_perturbed_score, \
               original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
               base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank


    def post_train(self,
                   kelpie_dataset: KelpieDataset,
                   kelpie_train_samples: numpy.array):
        """

        :param kelpie_dataset:
        :param kelpie_train_samples:
        :return:
        """
        kelpie_model_class = self.model.kelpie_model_class()
        kelpie_model = kelpie_model_class(model=self.model, dataset=kelpie_dataset)
        kelpie_model.to('cuda')

        optimizer = self.kelpie_optimizer_class(model=kelpie_model, hyperparameters=self.hyperparameters, verbose=False)
        optimizer.train(train_samples=kelpie_train_samples)
        return kelpie_model


    def extract_detailed_performances_on_sample(self,
                                                model: Model,
                                                sample: numpy.array):
        model.eval()
        head_id, relation_id, tail_id = sample

        # check how the model performs on the sample to explain
        all_scores = model.all_scores(numpy.array([sample])).detach().cpu().numpy()[0]
        target_entity_score = all_scores[tail_id] # todo: this only works in "head" perspective
        filter_out = model.dataset.to_filter[(head_id, relation_id)] if (head_id, relation_id) in model.dataset.to_filter else []

        if model.is_minimizer():
            all_scores[filter_out] = 1e6
            best_entity_score = numpy.min(all_scores)
            target_entity_rank = numpy.sum(all_scores <= target_entity_score)  # we use min policy here

        else:
            all_scores[filter_out] = -1e6
            best_entity_score = numpy.max(all_scores)
            target_entity_rank = numpy.sum(all_scores >= target_entity_score)  # we use min policy here

        return target_entity_score, best_entity_score, target_entity_rank
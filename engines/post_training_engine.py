import math
from typing import Tuple, Any

import numpy

from dataset import KelpieDataset, Dataset
from engines.engine import ExplanationEngine
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.optimization.bce_optimizer import KelpieBCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import KelpieMultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import KelpiePairwiseRankingOptimizer
from model import Model, KelpieModel
from collections import OrderedDict

class PostTrainingEngine(ExplanationEngine):

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        """
            PostTrainingEngine constructor.

            :param model: the trained Model to explain the behaviour of. This can NOT be a KelpieModel.
            :param dataset: the Dataset used to train the model
            :param hyperparameters: dict containing all the hyperparameters necessary for running the post-training
                                    (for both the model and the optimizer)
        """

        ExplanationEngine.__init__(self, model=model, dataset=dataset, hyperparameters=hyperparameters)


        if isinstance(self.model, ComplEx):
            self.kelpie_optimizer_class = KelpieMultiClassNLLOptimizer
        elif isinstance(self.model, ConvE):
            self.kelpie_optimizer_class = KelpieBCEOptimizer
        elif isinstance(self.model, TransE):
            self.kelpie_optimizer_class = KelpiePairwiseRankingOptimizer
        else:
            self.kelpie_optimizer_class = KelpieMultiClassNLLOptimizer

        if isinstance(model, KelpieModel):
            raise Exception("The model passed to the PostTrainingEngine is already a post-trainable KelpieModel.")

        # these data structures are used store permanently, for any fact:
        #   - the score
        #   - the score obtained by the best scoring tail (in "head" perspective) or head (in "tail" perspective)
        #   - the rank obtained by the target tail (in "head" perspective) or head (in "tail" perspective) score)
        self._original_model_results = {}  # map original samples to scores and ranks from the original model
        self._base_pt_model_results = {}   # map original samples to scores and ranks from the base post-trained model

        # The kelpie_cache is a simple LRU cache that allows reuse of KelpieDatasets and of base post-training results
        # without need to re-build them from scratch every time.
        self._kelpie_dataset_cache_size = 20
        self._kelpie_dataset_cache = OrderedDict()



    def simple_removal_explanations(self,
                                    sample_to_explain: Tuple[Any, Any, Any],
                                    perspective: str,
                                    top_k: int =-1):
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

            # we want to give higher priority to the facts that, when removed, worsen the rank and the score the most.
            rank_worsening = cur_tail_rank - base_tail_rank

            # if the model is a minimizer the greater cur_direct_score is than base_direct_score, the more relevant the removed fact;
            # if the model is a maximizer that lesser cur_direct_score is than base_direct_score, the more relevant the removed fact
            score_worsening = cur_direct_score - base_direct_score if self.model.is_minimizer() else base_direct_score - cur_direct_score
            relevance = float(rank_worsening + self.sigmoid(score_worsening))/float(cur_tail_rank)

            results.append((tuple(skipped_training_sample), cur_direct_score, cur_inverse_score, cur_head_rank, cur_tail_rank, relevance))

        results = sorted(results, key=lambda element: element[-1], reverse=True)

        if top_k == -1 or top_k < len(results):
            return results
        else:
            return results[:top_k]


    def addition_relevance(self,
                           sample_to_convert: Tuple[Any, Any, Any],
                           perspective: str,
                           samples_to_add: list):
        """
            Given a "sample to convert" (that is, a sample that the model currently does not predict as true,
            and that we want to be predicted as true);
            given the perspective from which to analyze it;
            and given and a list of other training samples containing the entity to convert;
            compute the relevance of the samples to add, that is, an estimate of the effect they would have
            if added (all together) to the perspective entity to improve the prediction of the sample to convert.

            :param sample_to_convert: the sample that we would like the model to predict as "true",
                                      in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param samples_to_add: the list of samples containing the perspective entity
                                   that we want to analyze the effect of, if added to the perspective entity

            :return:
        """

        head_id, relation_id, tail_id = sample_to_convert
        original_entity_to_convert = head_id if perspective == "head" else tail_id

        # check how the original model performs on the original sample to convert
        original_target_entity_score, \
        original_best_entity_score, \
        original_target_entity_rank = self.original_results_for(original_sample_to_predict=sample_to_convert)

        # get from the cache a Kelpie Dataset focused on the original id of the entity to explain,
        # (or create it from scratch if it is not in cache)
        kelpie_dataset = self._get_kelpie_dataset_for(original_entity_id=original_entity_to_convert)

        # run base post-training to obtain a "clone" of the perspective entity and see how it performs in the sample to convert
        base_pt_target_entity_score, \
        base_pt_best_entity_score, \
        base_pt_target_entity_rank = self.base_post_training_results_for(kelpie_dataset=kelpie_dataset,
                                                                         original_sample_to_predict=sample_to_convert)

        # run actual post-training by adding the passed samples to the perspective entity and see how it performs in the sample to convert
        pt_target_entity_score, \
        pt_best_entity_score, \
        pt_target_entity_rank = self.addition_post_training_results_for(kelpie_dataset=kelpie_dataset,
                                                                        original_sample_to_predict=sample_to_convert,
                                                                        original_samples_to_add=samples_to_add)

        # we want to give higher priority to the facts that, when added, make the score the better (= smaller).
        rank_improvement = base_pt_target_entity_rank - pt_target_entity_rank

        # if the model is a minimizer the smaller pt_target_entity_score is than base_pt_target_entity_score, the more relevant the added fact;
        # if the model is a maximizer, the greater pt_target_entity_score is than base_pt_target_entity_score, the more relevant the added fact
        score_improvement = base_pt_target_entity_score - pt_target_entity_score if self.model.is_minimizer() else pt_target_entity_score - base_pt_target_entity_score

        relevance = float(rank_improvement + self.sigmoid(score_improvement)) / float(base_pt_target_entity_rank)

        print("\t\tObtained individual relevance: " + str(relevance) + "\n")

        return relevance, \
               original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
               base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank,\
               pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank


    # private methods that know how to access cache structures

    def _get_kelpie_dataset_for(self, original_entity_id: int) -> KelpieDataset:
        """
        Return the value of the queried key in O(1).
        Additionally, move the key to the end to show that it was recently used.

        :param original_entity_id:
        :return:
        """

        if original_entity_id not in self._kelpie_dataset_cache:

            kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_id=original_entity_id)
            self._kelpie_dataset_cache[original_entity_id] = kelpie_dataset
            self._kelpie_dataset_cache.move_to_end(original_entity_id)

            if len(self._kelpie_dataset_cache) > self._kelpie_dataset_cache_size:
                self._kelpie_dataset_cache.popitem(last=False)

        return self._kelpie_dataset_cache[original_entity_id]

    def original_results_for(self, original_sample_to_predict: numpy.array) :

        sample = (original_sample_to_predict[0], original_sample_to_predict[1], original_sample_to_predict[2])
        if not sample in self._original_model_results:
            target_entity_score, \
            best_entity_score, \
            target_entity_rank = self.extract_detailed_performances_on_sample(self.model, original_sample_to_predict)

            self._original_model_results[sample] = (target_entity_score, best_entity_score, target_entity_rank)

        return self._original_model_results[sample]


    def base_post_training_results_for(self,
                                       kelpie_dataset: KelpieDataset,
                                       original_sample_to_predict: numpy.array):

        original_sample_to_predict = (original_sample_to_predict[0], original_sample_to_predict[1], original_sample_to_predict[2])
        kelpie_sample_to_predict = kelpie_dataset.as_kelpie_sample(original_sample=original_sample_to_predict)


        if not original_sample_to_predict in self._base_pt_model_results:
            original_entity_name = kelpie_dataset.entity_id_2_name[kelpie_dataset.original_entity_id]
            print("\t\tRunning base post-training on entity " + original_entity_name + " with no additions")
            base_pt_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                            kelpie_train_samples=kelpie_dataset.kelpie_train_samples) # type: KelpieModel

            # then check how the base post-trained model performs on the kelpie sample to explain.
            # This means checking how the "clone entity" (with no additional samples) performs
            base_pt_target_entity_score, \
            base_pt_best_entity_score, \
            base_pt_target_entity_rank = self.extract_detailed_performances_on_sample(base_pt_model, kelpie_sample_to_predict)

            self._base_pt_model_results[original_sample_to_predict] = (base_pt_target_entity_score,
                                                                       base_pt_best_entity_score,
                                                                       base_pt_target_entity_rank)

        return self._base_pt_model_results[original_sample_to_predict]


    def addition_post_training_results_for(self,
                                           kelpie_dataset: KelpieDataset,
                                           original_sample_to_predict: numpy.array,
                                           original_samples_to_add: numpy.array):

        original_sample_to_predict = (original_sample_to_predict[0], original_sample_to_predict[1], original_sample_to_predict[2])
        kelpie_sample_to_predict = kelpie_dataset.as_kelpie_sample(original_sample=original_sample_to_predict)

        # these are original samples, and not "kelpie" samples.
        # the "add_training_samples" method replaces the original entity with the kelpie entity by itself
        kelpie_dataset.add_training_samples(original_samples_to_add)

        original_entity_name = kelpie_dataset.entity_id_2_name[kelpie_dataset.original_entity_id]
        print("\t\tRunning post-training on entity " + original_entity_name + " adding samples: ")
        for x in original_samples_to_add:
            print ("\t\t\t" + kelpie_dataset.printable_sample(x))

        # post-train a kelpie model on the dataset that has undergone the addition
        cur_kelpie_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                           kelpie_train_samples=kelpie_dataset.kelpie_train_samples)  # type: KelpieModel

        # then check how the post-trained model performs on the kelpie sample to explain.
        # This means checking how the "kelpie entity" (with the added sample) performs, rather than the original entity
        pt_target_entity_score, \
        pt_best_entity_score, \
        pt_target_entity_rank = self.extract_detailed_performances_on_sample(cur_kelpie_model, kelpie_sample_to_predict)

        # undo the addition, to allow the following iterations of this loop
        kelpie_dataset.undo_last_training_samples_addition()

        return pt_target_entity_score, pt_best_entity_score, pt_target_entity_rank


    # private methods to do stuff

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

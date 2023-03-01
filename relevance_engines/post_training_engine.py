import math
import time
from typing import Tuple, Any
import numpy
import torch

from dataset import Dataset
from kelpie_dataset import KelpieDataset
from relevance_engines.engine import ExplanationEngine
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.models.tucker import TuckER
from link_prediction.optimization.bce_optimizer import KelpieBCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import KelpieMultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import KelpiePairwiseRankingOptimizer
from link_prediction.models.model import *
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
        self._kelpie_dataset_cache_size = kelpie_dataset_cache_size
        self._kelpie_dataset_cache = OrderedDict()
        self.print_count = 0
        self.original_sample = None
        self.kelpie_dataset = None

    def addition_relevance(self,
                           sample_to_convert: Tuple[Any, Any, Any],
                           perspective: str,
                           samples_to_add: list):
        """
            Given a "sample to convert" (that is, a sample that the model currently does not predict as true,
            and that we want to be predicted as true);
            given the perspective from which to analyze it;
            and given and a list of samples containing the entity to convert and that were not seen in training;
            compute the relevance of the samples to add, that is, an estimate of the effect they would have
            if added (all together) to the perspective entity to improve the prediction of the sample to convert.

            :param sample_to_convert: the sample that we would like the model to predict as "true",
                                      in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param samples_to_add: the list of samples containing the perspective entity
                                   that we want to analyze the effect of, if added to the perspective entity

            :return:
        """
        start_time = time.time()

        self.original_sample = sample_to_convert
        # check how the original model performs on the original sample to convert
        original_target_entity_score, \
        original_best_entity_score, \
        original_target_entity_rank = self.original_results_for(original_sample_to_predict=sample_to_convert)

        kelpie_dataset = self.get_kelpie_dataset(sample_to_convert, perspective)

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
        score_improvement = base_pt_target_entity_score - pt_target_entity_score if self.model.is_minimizer() else pt_target_entity_score - base_pt_target_entity_score

        relevance = float(rank_improvement + self.sigmoid(score_improvement)) / float(base_pt_target_entity_rank)
        relevance = rd(relevance)

        return relevance, \
               original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
               base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank,\
               pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, \
               rd(time.time() - start_time)
    
    def get_kelpie_dataset(self, sample_to_explain: Tuple[Any, Any, Any], perspective: str):
        # get from the cache a Kelpie Dataset focused on the original id of the entity to explain,
        # (or create it from scratch if it is not in cache)
        head_id, _, tail_id = sample_to_explain
        if global_dic['args'].relation_path:
            return self._get_kelpie_dataset_for(entity_ids=[head_id, tail_id])
        
        original_entity_to_convert = head_id if perspective == "head" else tail_id
        return self._get_kelpie_dataset_for(entity_ids=[original_entity_to_convert])


    def removal_relevance(self,
                           sample_to_explain: Tuple[Any, Any, Any],
                           perspective: str,
                           samples_to_remove: list):
        """
            Given a "sample to explain" (that is, a sample that the model currently predicts as true,
            and that we want to be predicted as false);
            given the perspective from which to analyze it;
            and given and a list of training samples containing the entity to convert;
            compute the relevance of the samples in removal, that is, an estimate of the effect they would have
            if removed (all together) from the perspective entity to worsen the prediction of the sample to convert.

            :param sample_to_explain: the sample that we would like the model to predict as "true",
                                      in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param samples_to_remove:   the list of samples containing the perspective entity
                                        that we want to analyze the effect of, if added to the perspective entity
        """
        print(f'removal relevance of sample: {str(sample_to_explain)}, removing: {str(samples_to_remove)}')
        start_time = time.time()

        self.original_sample = sample_to_explain
        self.kelpie_dataset = self.get_kelpie_dataset(sample_to_explain, perspective)

        # check how the original model performs on the original sample to convert (no need)
        original_target_entity_score, \
        original_target_entity_rank, \
        original_best_entity_score = self.original_results_for()

        # get from the cache a Kelpie Dataset focused on the original id of the entity to explain,
        # (or create it from scratch if it is not in cache)

        base_pt_target_entity_score, \
        base_pt_target_entity_rank, \
        base_pt_best_entity_score= self.base_post_training_results_for()

        # run actual post-training by adding the passed samples to the perspective entity and see how it performs in the sample to convert
        pt_target_entity_score, \
        pt_target_entity_rank, \
        pt_best_entity_score,= self.removal_post_training_results_for(original_samples_to_remove=samples_to_remove)

        # we want to give higher priority to the facts that, when added, make the score worse (= higher).
        rank_worsening = pt_target_entity_rank - base_pt_target_entity_rank
        score_worsening = pt_target_entity_score - base_pt_target_entity_score if self.model.is_minimizer() else base_pt_target_entity_score - pt_target_entity_score

        # note: the formulation is very different from the addition one
        relevance = rd(float(rank_worsening + self.sigmoid(score_worsening)))

        return relevance, \
               original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
               base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank,\
               pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, \
               rd(time.time() - start_time)



    # private methods that know how to access cache structures

    def _get_kelpie_dataset_for(self, entity_ids) -> KelpieDataset:
        """
        Return the value of the queried key in O(1).
        Additionally, move the key to the end to show that it was recently used.

        :param original_entity_id:
        :return:
        """

        name = strfy(entity_ids)
        if name not in self._kelpie_dataset_cache:

            kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_ids=entity_ids)
            self._kelpie_dataset_cache[name] = kelpie_dataset
            self._kelpie_dataset_cache.move_to_end(name)

            if len(self._kelpie_dataset_cache) > self._kelpie_dataset_cache_size:
                self._kelpie_dataset_cache.popitem(last=False)

        return self._kelpie_dataset_cache[name]

    def original_results_for(self) :
        sample = self.original_sample
        if not sample in self._original_model_results:
            self._original_model_results[sample] = self.extract_detailed_performances_on_sample(self.model, sample)

        return self._original_model_results[sample]


    def base_post_training_results_for(self):

        """
        :param kelpie_dataset:
        :param original_sample_to_predict:
        :return:
        """
        if self.original_sample in self._base_pt_model_results:   # cache
            return self._base_pt_model_results[self.original_sample]
        
        # kelpie_model: an UNTRAINED kelpie model that has just been initialized
        kelpie_model_class = self.model.kelpie_model_class()
        kelpie_model = kelpie_model_class(model=self.model, dataset=self.kelpie_dataset)

        kelpie_sample_to_predict = self.kelpie_dataset.as_kelpie_sample(original_sample=self.original_sample)

        kelpie_model.summary('before post_train')
        # base_pt_model = kelpie_model
        base_pt_model = self.post_train(kelpie_model_to_post_train=kelpie_model,
                                        kelpie_train_samples=self.kelpie_dataset.kelpie_train_samples) # type: KelpieModel
        kelpie_model.summary('after post_train')

        # then check how the base post-trained model performs on the kelpie sample to explain.
        # This means checking how the "clone entity" (with no additional samples) performs
        self._base_pt_model_results[self.original_sample] = self.extract_detailed_performances_on_sample(base_pt_model, kelpie_sample_to_predict)

        return self._base_pt_model_results[self.original_sample]


    def addition_post_training_results_for(self,
                                           kelpie_model: KelpieModel,
                                           kelpie_dataset: KelpieDataset,
                                           original_sample_to_predict: numpy.array,
                                           original_samples_to_add: numpy.array):

        """

        :param kelpie_model: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_dataset:
        :param original_sample_to_predict:
        :param original_samples_to_add:
        :return:
        """
        kelpie_sample_to_predict = kelpie_dataset.as_kelpie_sample(original_sample=original_sample_to_predict)
        # these are original samples, and not "kelpie" samples.
        # the "add_training_samples" method replaces the original entity with the kelpie entity by itself
        kelpie_dataset.add_training_samples(original_samples_to_add)

        # original_entity_name = kelpie_dataset.entity_id_2_name[kelpie_dataset.original_entity_id]
        print(f"\t\t\tAdding samples: ", end='')
        for x in original_samples_to_add:
            print (kelpie_dataset.printable_sample(x), end=',')

        # post-train the kelpie model on the dataset that has undergone the addition
        cur_kelpie_model = self.post_train(kelpie_model_to_post_train=kelpie_model,
                                           kelpie_train_samples=kelpie_dataset.kelpie_train_samples)  # type: KelpieModel

        kelpie_dataset.undo_last_training_samples_addition()

        # then check how the post-trained model performs on the kelpie sample to explain.
        # This means checking how the "kelpie entity" (with the added sample) performs, rather than the original entity
        return self.extract_detailed_performances_on_sample(cur_kelpie_model, kelpie_sample_to_predict)


    def removal_post_training_results_for(self, original_samples_to_remove: numpy.array):
        """
        :param kelpie_dataset:
        :param original_sample_to_predict:
        :param original_samples_to_remove:
        :return:
        """
        if global_dic['args'].relation_path:
            print('\tpaths:', original_samples_to_remove)
            tmp = set()
            # 共同路径头/尾
            num = self.dataset.num_direct_relations
            for p in original_samples_to_remove:    # remove samples connected to head/tail
                tmp.add(get_forward_sample(p[0], num))
                tmp.add(get_forward_sample(p[-1], num))
            original_samples_to_remove = tmp
        print('\tremoving samples:', original_samples_to_remove)

        # kelpie_model: an UNTRAINED kelpie model that has just been initialized
        kelpie_model_class = self.model.kelpie_model_class()
        kelpie_model = kelpie_model_class(model=self.model, dataset=self.kelpie_dataset)

        kelpie_sample_to_predict = self.kelpie_dataset.as_kelpie_sample(original_sample=self.original_sample)

        # these are original samples, and not "kelpie" samples.
        # the "remove_training_samples" method replaces the original entity with the kelpie entity by itself
        self.kelpie_dataset.remove_training_samples(original_samples_to_remove)

        # post-train a kelpie model on the dataset that has undergone the removal
        kelpie_model.summary('before post_train')
        # cur_kelpie_model = kelpie_model
        cur_kelpie_model = self.post_train(kelpie_model_to_post_train=kelpie_model,
                                           kelpie_train_samples=self.kelpie_dataset.kelpie_train_samples)  # type: KelpieModel
        kelpie_model.summary('after post_train')

        # undo the removal, to allow the following iterations of this loop
        self.kelpie_dataset.undo_last_training_samples_removal()

        # checking how the "kelpie entity" (without the removed samples) performs, rather than the original entity
        return self.extract_detailed_performances_on_sample(cur_kelpie_model, kelpie_sample_to_predict)


    # private methods to do stuff
    def post_train(self,
                   kelpie_model_to_post_train: KelpieModel,
                   kelpie_train_samples: numpy.array):
        """

        :param kelpie_model_to_post_train: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_train_samples:
        :return:
        """
        # kelpie_model_class = self.model.kelpie_model_class()
        # kelpie_model = kelpie_model_class(model=self.model, dataset=kelpie_dataset)
        kelpie_model_to_post_train.to('cuda')

        optimizer = self.kelpie_optimizer_class(model=kelpie_model_to_post_train,
                                                hyperparameters=self.hyperparameters,
                                                verbose=False)
        optimizer.epochs = self.hyperparameters[RETRAIN_EPOCHS]
        t = time.time()
        optimizer.train(train_samples=kelpie_train_samples)
        if self.print_count < 5:
            self.print_count += 1
            print(f'\t\t[post_train_time: {rd(time.time() - t)}]')
        return kelpie_model_to_post_train


    def extract_detailed_performances_on_sample(self,
                                                model: Model,
                                                sample: numpy.array):
        
        # return model.predict_tail(sample)
        
        print('evaluating', sample)
        model.eval()
        head_id, relation_id, tail_id = sample

        # check how the model performs on the sample to explain
        all_scores = model.all_scores(numpy.array([sample]), sigmoid=False).detach().cpu().numpy()[0]

        print('original target score:', all_scores[[self.original_sample[0], self.original_sample[2]]])

        print('all score:', all_scores)
        target_entity_score = all_scores[tail_id] # todo: this only works in "head" perspective
        filter_out = model.dataset.to_filter[(head_id, relation_id)] if (head_id, relation_id) in model.dataset.to_filter else []

        if model.is_minimizer():
            all_scores[filter_out] = 1e6
            # if the target score had been filtered out, put it back
            # (this may happen in necessary mode, where we may run this method on the actual test sample;
            # not in sufficient mode, where we run this method on the unseen "samples to convert")
            all_scores[tail_id] = target_entity_score
            best_entity_score = numpy.min(all_scores)
            target_entity_rank = numpy.sum(all_scores <= target_entity_score)  # we use min policy here

        else:
            all_scores[filter_out] = -1e6
            # if the target score had been filtered out, put it back
            # (this may happen in necessary mode, where we may run this method on the actual test sample;
            # not in sufficient mode, where we run this method on the unseen "samples to convert")
            all_scores[tail_id] = target_entity_score
            best_entity_score = numpy.max(all_scores)
            target_entity_rank = numpy.sum(all_scores >= target_entity_score)  # we use min policy here

        print('target_entity_score', rd(target_entity_score))
        print('best_entity_score', rd(best_entity_score))
        print('target_entity_rank', target_entity_rank)

        return rd(target_entity_score), target_entity_rank, rd(best_entity_score)

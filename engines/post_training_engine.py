import copy
from typing import Type, Tuple, Any

import numpy
import torch

from dataset import KelpieDataset, Dataset
from engines.engine import ExplanationEngine
from link_prediction.evaluation.evaluation import KelpieEvaluator
from link_prediction.perturbation import kelpie_perturbation
from model import Model, KelpieModel

class PostTrainingEngine(ExplanationEngine):

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

        self.model = model
        self.kelpie_optimizer_class = post_training_optimizer_class
        self.model.to('cuda')   # it this hasn't been done yet, load the model in GPU
        self.dataset = dataset
        self.hyperparameters = hyperparameters

    def simple_removal_explanations(self,
                                    sample_to_explain: Tuple[Any, Any, Any],
                                    perspective: str,
                                    top_k: int):

        # the behaviour of the engine must be deterministic!
        seed = 42
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True

        # sample to explain as a numpy array of the ids
        head_id, relation_id, tail_id = sample_to_explain
        # name and id of the entity to explain, based on the passed perspective
        original_entity_id = head_id if perspective == "head" else tail_id

        # create a Kelpie Dataset focused on the original id of the entity to explain
        kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_id=original_entity_id)

        print("\tEntity to explain: " + self.dataset.entity_id_2_name[original_entity_id] + " (degree: " + str(self.dataset.entity_2_degree[original_entity_id]) + ")")
        print("\tRunning base post-training before removal...")
        post_trained_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                             kelpie_train_samples=kelpie_dataset.kelpie_train_samples) # type: KelpieModel

        # check how the post-trained "clone" performs on the sample to explain
        #direct_score, inverse_score, head_rank, tail_rank = self.test_kelpie_model(testing_model=post_trained_model, sample_to_explain=sample_to_explain, perspective=perspective)
        (direct_score, inverse_score), _, _ = post_trained_model.predict_sample(sample=numpy.array(sample_to_explain),
                                                                                    original_mode=True)

        # TODO: make interfaces uniform, that is, make all components return either samples with kelpie entity id or samples with original entity id
        # right now, some (e.g. kelpie_perturbation.perturbate_samples) return samples with kelpie_entity_id
        # and some others require or return samples with original_entity_id (e.g. test_kelpie_model_on_sample)
        # this is confusing!

        # introduce perturbation
        perturbed_list, skipped_list = kelpie_perturbation.perturbate_samples(kelpie_dataset.kelpie_train_samples)

        results = []
        # for each training sample, run a post-training skipping that sample
        for i in range(len(perturbed_list)):
            cur_training_samples = perturbed_list[i]

            cur_skipped_sample = Dataset.replace_entity_in_sample(sample=skipped_list[i][0],
                                                                  old_entity=kelpie_dataset.kelpie_entity_id,
                                                                  new_entity=kelpie_dataset.original_entity_id)

            head_name = kelpie_dataset.entity_id_2_name[int(cur_skipped_sample[0])]
            rel_name = kelpie_dataset.relation_id_2_name[int(cur_skipped_sample[1])]
            tail_name = kelpie_dataset.entity_id_2_name[int(cur_skipped_sample[2])]

            print("\tIteration " + str(i) + ": skipping sample <"+ ", ".join([head_name, rel_name, tail_name]) + ">")

            cur_kelpie_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                               kelpie_train_samples=cur_training_samples)
            (cur_direct_score, cur_inverse_score), \
            (cur_head_rank, cur_tail_rank), _ = self.test_kelpie_model_on_sample(model=cur_kelpie_model,
                                                                                 sample=numpy.array(sample_to_explain))

            direct_diff, inverse_diff = direct_score-cur_direct_score, inverse_score-cur_inverse_score

            results.append((cur_skipped_sample, cur_direct_score, cur_inverse_score, cur_head_rank, cur_tail_rank, direct_diff + inverse_diff))

        results = sorted(results, key=lambda element: element[-1], reverse=True)

        if top_k == -1 or top_k < len(results):
            return results
        else:
            return results[:top_k]


    def simple_addition_explanations(self,
                    sample_to_explain: Tuple[Any, Any, Any],
                    perspective: str,
                    samples_to_add: list):

        # sample to explain as a numpy array of the ids
        head_id, relation_id, tail_id = sample_to_explain
        # name and id of the entity to explain, based on the passed perspective
        original_entity_id = head_id if perspective == "head" else tail_id

        # create a Kelpie Dataset focused on the original id of the entity to explain
        kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_id=original_entity_id)

        print("\t\tRunning base post-training before additions...")
        base_pt_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                             kelpie_train_samples=kelpie_dataset.kelpie_train_samples) # type: KelpieModel

        # check how the
        # check how the post-trained "clone" performs on the sample to explain
        original_target_entity_score, \
        original_best_entity_score, \
        original_target_entity_rank = self.extract_detailed_performances_on_sample(self.model, sample_to_explain)

        base_pt_target_entity_score, \
        base_pt_best_entity_score, \
        base_pt_target_entity_rank = self.extract_detailed_performances_on_sample(base_pt_model, sample_to_explain)


        added_sample_2_relevance = dict()
        added_sample_2_pt_results = dict()
        for i in range(len(samples_to_add)):
            cur_sample_to_add = samples_to_add[i]

            head_name = kelpie_dataset.entity_id_2_name[int(cur_sample_to_add[0])]
            rel_name = kelpie_dataset.relation_id_2_name[int(cur_sample_to_add[1])]
            tail_name = kelpie_dataset.entity_id_2_name[int(cur_sample_to_add[2])]
            print("\t\tIteration " + str(i) + ": adding sample <"+ ", ".join([head_name, rel_name, tail_name]) + ">")

            # create a Kelpie Dataset that also contains the fact to add
            cur_kelpie_dataset = copy.deepcopy(kelpie_dataset)

            cur_sample_to_add_numpy_arr = numpy.array([cur_sample_to_add])
            cur_kelpie_dataset.add_training_samples(cur_sample_to_add_numpy_arr)

            cur_kelpie_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                               kelpie_train_samples=cur_kelpie_dataset.kelpie_train_samples)  # type: KelpieModel

            pt_target_entity_score, \
            pt_best_entity_score, \
            pt_target_entity_rank = self.extract_detailed_performances_on_sample(cur_kelpie_model, sample_to_explain)

            # TODO: this should depend on maximize/minimize
            direct_diff = pt_target_entity_score - base_pt_target_entity_score

            added_sample_2_relevance[cur_sample_to_add] = direct_diff
            added_sample_2_pt_results[cur_sample_to_add] = (pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank)

        sorted_samples_with_relevance = sorted(added_sample_2_relevance.items(), key=lambda element: element[-1], reverse=True)

        return sorted_samples_with_relevance, \
               added_sample_2_pt_results, \
               original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
               base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank

    def nple_addition_explanations(self,
                    sample_to_explain: Tuple[Any, Any, Any],
                    perspective: str,
                    samples_to_add: list,
                                   n: int):

        # if there are less than n items in samples_to_add
        # it is not possible to try combinations of length n of samples_to_add
        if len(samples_to_add) < n:
            return []

        # the behaviour of the engine must be deterministic!
        seed = 42
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True

        # identify the entity to explain
        head_id, relation_id, tail_id = sample_to_explain
        original_entity_id = head_id if perspective == "head" else tail_id

        # create a Kelpie Dataset focused on the original id of the entity to explain
        kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_id=original_entity_id)

        print("### BASE POST-TRAINING")
        post_trained_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                             kelpie_train_samples=kelpie_dataset.kelpie_train_samples) # type: KelpieModel

        # check how the post-trained "clone" performs on the sample to explain
        all_scores = post_trained_model.all_scores(numpy.array([sample_to_explain])).detach().cpu().numpy()[0]
        target_entity_score = all_scores[tail_id]
        filter_out = self.dataset.to_filter[(head_id, relation_id)]
        all_scores[filter_out] = -1e6  # todo: make it huge if the model wants to minimize scores
        best_entity_score = numpy.max(all_scores)  # todo: min if the model wants to minimize scores


        # extract all n-long different combinations of samples
        samples_nples = self._extract_sample_nples(samples=samples_to_add, n=n)

        added_nple_2_perturbed_score = {}
        added_nple_2_relevance = {}

        for i in range(len(samples_nples)):
            current_nple_to_add = samples_nples[i]

            # create a Kelpie Dataset ... that also contains the facts in the nple to add
            cur_kelpie_dataset = copy.deepcopy(kelpie_dataset)

            cur_nple_to_add_numpy_arr = numpy.array(current_nple_to_add)
            cur_kelpie_dataset.add_training_samples(cur_nple_to_add_numpy_arr)

            cur_kelpie_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                               kelpie_train_samples=cur_kelpie_dataset.kelpie_train_samples)  # type: KelpieModel

            (cur_direct_score, cur_inverse_score), _, _ = self.test_kelpie_model_on_sample(model=cur_kelpie_model,
                                                                                 sample=numpy.array(sample_to_explain))

            # TODO: this should depend on maximize/minimize
            direct_diff = cur_direct_score - target_entity_score

            added_nple_2_relevance[current_nple_to_add] = direct_diff
            added_nple_2_perturbed_score[current_nple_to_add] = cur_direct_score

        sorted_nples_with_relevance = sorted(added_nple_2_relevance.items(), key=lambda element: element[-1], reverse=True)

        return sorted_nples_with_relevance, \
               added_nple_2_perturbed_score, \
               target_entity_score, \
               best_entity_score


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


    def test_kelpie_model_on_sample(self,
                          model: KelpieModel,
                          sample: numpy.array):

        # convert the sample into a "kelpie sample", that is,
        # transform all occurrences of the original entity into the kelpie entity
        kelpie_sample = Dataset.replace_entity_in_sample(sample=sample,
                                                         old_entity=model.original_entity_id,
                                                         new_entity=model.kelpie_entity_id)
        # results on kelpie fact
        scores, ranks, predictions = model.predict_sample(sample=kelpie_sample, original_mode=False)
        return scores, ranks, predictions


    def extract_detailed_performances_on_sample(self,
                                                model: Model,
                                                sample: numpy.array):
        model.eval()
        head_id, relation_id, tail_id = sample
        # check how the post-trained "clone" performs on the sample to explain
        all_scores = model.all_scores(numpy.array([sample])).detach().cpu().numpy()[0]
        target_entity_score = all_scores[tail_id] # todo: this only works in "head" perspective
        filter_out = model.dataset.to_filter[(head_id, relation_id)]
        all_scores[filter_out] = -1e6  # todo: make it huge if the model wants to minimize scores
        best_entity_score = numpy.max(all_scores)  # todo: min if the model wants to minimize scores

        # todo: MIN POLICY HERE
        # todo: <= if the model wants to minimize scores
        target_entity_rank = numpy.sum(all_scores >= target_entity_score)

        return target_entity_score, best_entity_score, target_entity_rank
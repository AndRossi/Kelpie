from multiprocessing.pool import ThreadPool as Pool
from typing import Tuple, Any
from dataset import Dataset
from link_prediction.models.model import Model
from prefilters.prefilter import PreFilter
import numpy as np
from config import MAX_PROCESSES
import threading

class TypeBasedPreFilter(PreFilter):
    """
    The TypeBasedPreFilter object is a PreFilter that relies on the entity types
    to extract the most promising samples for an explanation.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset):
        """
        TypeBasedPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

        self.entity_id_2_train_samples = {}
        self.entity_id_2_relation_vector = {}

        self.threadLock = threading.Lock()
        self.counter = 0
        self.thread_pool = Pool(processes=MAX_PROCESSES)

        for (h, r, t) in dataset.train_samples:

            if not h in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[h] = [(h, r, t)]
                self.entity_id_2_relation_vector[h] = np.zeros(2*self.dataset.num_relations)

            if not t in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[t] = [(h, r, t)]
                self.entity_id_2_relation_vector[t] = np.zeros(2*self.dataset.num_relations)

            self.entity_id_2_train_samples[h].append((h, r, t))
            self.entity_id_2_train_samples[t].append((h, r, t))
            self.entity_id_2_relation_vector[h][r] += 1
            self.entity_id_2_relation_vector[t][r+self.dataset.num_relations] += 1


    def top_promising_samples_for(self,
                                  sample_to_explain:Tuple[Any, Any, Any],
                                  perspective:str,
                                  top_k=50,
                                  verbose=True):

        """
        This method extracts the top k promising samples for interpreting the sample to explain,
        from the perspective of either its head or its tail (that is, either featuring its head or its tail).

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the explanation perspective. It can be either "head" or "tail":
                                - if "head", find the most promising samples featuring the head of the sample to explain
                                - if "tail", find the most promising samples featuring the tail of the sample to explain
        :param top_k: the number of top promising samples to extract.
        :param verbose:
        :return: the sorted list of the k most promising samples.
        """
        self.counter = 0

        if verbose:
            print("Type-based extraction of promising facts for" + self.dataset.printable_sample(sample_to_explain))

        head, relation, tail = sample_to_explain

        if perspective == "head":
            samples_featuring_head = self.entity_id_2_train_samples[head]

            worker_processes_inputs = [(len(samples_featuring_head), sample_to_explain, x, perspective, verbose) for x
                                       in samples_featuring_head]
            results = self.thread_pool.map(self._analyze_sample, worker_processes_inputs)

            sample_featuring_head_2_promisingness = {}

            for i in range(len(samples_featuring_head)):
                sample_featuring_head_2_promisingness[samples_featuring_head[i]] = results[i]

            samples_featuring_head_with_promisingness = sorted(sample_featuring_head_2_promisingness.items(),
                                                               reverse=True,
                                                               key=lambda x: x[1])
            sorted_promising_samples = [x[0] for x in samples_featuring_head_with_promisingness]

        else:
            samples_featuring_tail = self.entity_id_2_train_samples[tail]

            worker_processes_inputs = [(len(samples_featuring_tail), sample_to_explain, x, perspective, verbose) for x
                                       in samples_featuring_tail]
            results = self.thread_pool.map(self._analyze_sample, worker_processes_inputs)

            sample_featuring_tail_2_promisingness = {}

            for i in range(len(samples_featuring_tail)):
                sample_featuring_tail_2_promisingness[samples_featuring_tail[i]] = results[i]

            samples_featuring_tail_with_promisingness = sorted(sample_featuring_tail_2_promisingness.items(),
                                                               reverse=True,
                                                               key=lambda x: x[1])
            sorted_promising_samples = [x[0] for x in samples_featuring_tail_with_promisingness]

        return sorted_promising_samples[:top_k]


    def _analyze_sample(self, input_data):
        all_samples_number, sample_to_explain, sample_to_analyze, perspective, verbose = input_data

        with self.threadLock:
            self.counter+=1
            i = self.counter

        if verbose:
            print("\tAnalyzing sample " + str(i) + " on " + str(all_samples_number) + ": " + self.dataset.printable_sample(sample_to_analyze))

        head_to_explain, relation_to_explain, tail_to_explain = sample_to_explain
        head_to_analyze, relation_to_analyze, tail_to_analyze = sample_to_analyze

        promisingness = -1
        if perspective == "head":
            if head_to_explain == head_to_analyze:
                promisingness = self._cosine_similarity(tail_to_explain, tail_to_analyze)
            else:
                assert(head_to_explain == tail_to_analyze)
                promisingness = self._cosine_similarity(tail_to_explain, head_to_analyze)

        elif perspective == "tail":
            if tail_to_explain == tail_to_analyze:
                promisingness = self._cosine_similarity(head_to_explain, head_to_analyze)
            else:
                assert (tail_to_explain == head_to_analyze)
                promisingness = self._cosine_similarity(head_to_explain, tail_to_analyze)

        return promisingness

    def _cosine_similarity(self, entity1_id, entity2_id):
        entity1_vector = self.entity_id_2_relation_vector[entity1_id]
        entity2_vector = self.entity_id_2_relation_vector[entity2_id]
        return np.inner(entity1_vector, entity2_vector) / (np.linalg.norm(entity1_vector) * np.linalg.norm(entity2_vector))
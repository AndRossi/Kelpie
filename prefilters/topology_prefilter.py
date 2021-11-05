import copy
from multiprocessing.pool import ThreadPool as Pool
from typing import Tuple, Any
from dataset import Dataset
from link_prediction.models.model import Model
from prefilters.prefilter import PreFilter

from config import MAX_PROCESSES
import threading

class TopologyPreFilter(PreFilter):
    """
    The TopologyPreFilter object is a PreFilter that relies on the graph topology
    to extract the most promising samples for an explanation.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset):
        """
        PostTrainingPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

        self.max_path_length = 5
        self.entity_id_2_train_samples = {}
        self.threadLock = threading.Lock()
        self.counter = 0
        self.thread_pool = Pool(processes=MAX_PROCESSES)

        for (h, r, t) in dataset.train_samples:

            if h in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[h].append((h, r, t))
            else:
                self.entity_id_2_train_samples[h] = [(h, r, t)]

            if t in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[t].append((h, r, t))
            else:
                self.entity_id_2_train_samples[t] = [(h, r, t)]


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
        :return: the sorted list of the k most promising samples.
        """
        self.counter = 0

        head, relation, tail = sample_to_explain

        if verbose:
            print("Extracting promising facts for" + self.dataset.printable_sample(sample_to_explain))

        start_entity, end_entity = (head, tail) if perspective == "head" else (tail, head)

        samples_featuring_start_entity = self.entity_id_2_train_samples[start_entity]

        sample_to_analyze_2_min_path_length = {}
        sample_to_analyze_2_min_path = {}

        worker_processes_inputs = [(len(samples_featuring_start_entity),
                                   start_entity, end_entity, samples_featuring_start_entity[i], verbose)
                                   for i in range(len(samples_featuring_start_entity))]

        results = self.thread_pool.map(self.analyze_sample, worker_processes_inputs)

        for i in range(len(samples_featuring_start_entity)):
            _, _, _, sample_to_analyze, _ = worker_processes_inputs[i]
            shortest_path_lengh, shortest_path = results[i]

            sample_to_analyze_2_min_path_length[sample_to_analyze] = shortest_path_lengh
            sample_to_analyze_2_min_path[sample_to_analyze] = shortest_path

        results = sorted(sample_to_analyze_2_min_path_length.items(), key=lambda x:x[1])
        results = [x[0] for x in results]

        return results[:top_k]

    def analyze_sample(self, input_data):
        all_samples_number, start_entity, end_entity, sample_to_analyze, verbose = input_data

        with self.threadLock:
            self.counter+=1
            i = self.counter

        if verbose:
            print("\tAnalyzing sample " + str(i) + " on " + str(all_samples_number) + ": " + self.dataset.printable_sample(sample_to_analyze))

        sample_to_analyze_head, sample_to_analyze_relation, sample_to_analyze_tail = sample_to_analyze

        cur_path_length = 1
        next_step_incomplete_paths = []   # each incomplete path is a couple (list of triples in this path, accretion entity)

        # if the sample to analyze is already a path from the start entity to the end entity,
        # then the shortest path length is 1 and you can move directly to the next sample to analyze
        if (sample_to_analyze_head == start_entity and sample_to_analyze_tail == end_entity) or \
                (sample_to_analyze_tail == start_entity and sample_to_analyze_head == end_entity):
            return cur_path_length, \
                   [(sample_to_analyze_head, sample_to_analyze_relation, sample_to_analyze_tail)]


        initial_accretion_entity = sample_to_analyze_tail if sample_to_analyze_head == start_entity else sample_to_analyze_head
        next_step_incomplete_paths.append(([sample_to_analyze], initial_accretion_entity))

        # this set contains the entities seen so far in the search.
        # we want to avoid any loops / unnecessary searches, so it is not allowed for a path
        # to visit an entity that has already been featured by another path
        # (that is, another path that has either same or smaller size!)
        entities_seen_so_far = {start_entity, initial_accretion_entity}

        terminate = False
        while not terminate:
            cur_path_length += 1

            cur_step_incomplete_paths = next_step_incomplete_paths
            next_step_incomplete_paths = []

            #print("\t\tIncomplete paths of length " + str(cur_path_length - 1) + " to analyze: " + str(len(cur_step_incomplete_paths)))
            #print("\t\tExpanding them to length: " + str(cur_path_length))
            for (incomplete_path, accretion_entity) in cur_step_incomplete_paths:
                samples_featuring_accretion_entity = self.entity_id_2_train_samples[accretion_entity]

                # print("\tCurrent path: " + str(incomplete_path))

                for (cur_head, cur_rel, cur_tail) in samples_featuring_accretion_entity:

                    cur_incomplete_path = copy.deepcopy(incomplete_path)

                    # print("\t\tCurrent accretion path: " + self.dataset.printable_sample((cur_h, cur_r, cur_t)))
                    if (cur_head == accretion_entity and cur_tail == end_entity) or (cur_tail == accretion_entity and cur_head == end_entity):
                        cur_incomplete_path.append((cur_head, cur_rel, cur_tail))
                        return cur_path_length, cur_incomplete_path

                    # ignore self-loops
                    if cur_head == cur_tail:
                        # print("\t\t\tMeh, it was just a self-loop!")
                        continue

                    # ignore facts that would re-connect to an entity that is already in this path
                    next_step_accretion_entity = cur_tail if cur_head == accretion_entity else cur_head
                    if next_step_accretion_entity in entities_seen_so_far:
                        # print("\t\t\tMeh, it led to a loop in this path!")
                        continue

                    cur_incomplete_path.append((cur_head, cur_rel, cur_tail))
                    next_step_incomplete_paths.append((cur_incomplete_path, next_step_accretion_entity))
                    entities_seen_so_far.add(next_step_accretion_entity)
                    # print("\t\t\tThe search continues")

            if terminate is not True:
                if cur_path_length == self.max_path_length or len(next_step_incomplete_paths) == 0:
                    return 1e6, ["None"]

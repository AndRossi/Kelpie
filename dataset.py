import copy
import html
import os
from collections import defaultdict
from typing import Tuple

import numpy

from config import ROOT

DATA_PATH = os.path.join(ROOT, "data")
FB15K = "FB15k"
FB15K_237 = "FB15k-237"
WN18 = "WN18"
WN18RR = "WN18RR"
YAGO3_10 = "YAGO3-10"

ALL_DATASET_NAMES = [FB15K, FB15K_237, WN18, WN18RR, YAGO3_10]

# relation types
ONE_TO_ONE="1-1"
ONE_TO_MANY="1-N"
MANY_TO_ONE="N-1"
MANY_TO_MANY="N-N"

class Dataset:

    def __init__(self,
                 name: str,
                 separator: str = "\t",
                 load: bool = True):
        """
            Dataset constructor.
            This method will initialize the Dataset and its structures.
            If parameter "load" is set to true, it will immediately read the dataset files
            and fill the data structures with the read data.
            :param name: the dataset name. It must correspond to the dataset folder name in DATA_PATH
            :param separator: the character that separates head, relation and tail in each triple in the dataset files
            :param load: boolean flag; if True, the dataset files must be accessed and read immediately.
        """

        # note: the "load" flag is necessary because the Kelpie datasets do not require loading,
        #       as they are built from already loaded pre-existing datasets.

        self.name = name
        self.separator = separator
        self.home = os.path.join(DATA_PATH, self.name)

        # train, valid and test set paths
        self.train_path = os.path.join(self.home, "train.txt")
        self.valid_path = os.path.join(self.home, "valid.txt")
        self.test_path = os.path.join(self.home, "test.txt")

        ### All these data structures are read from filesystem lazily using the load method ###

        # sets of entities and relations in their textual format
        self.entities, self.relations = set(), set()

        # maps that associate to each entity/relation name (id) the corresponding entity/relation id (name)
        self.entity_name_2_id, self.entity_id_2_name = dict(), dict()
        self.relation_name_2_id, self.relation_id_2_name = dict(), dict()

        # collections of triples (facts with textual names) and samples (facts with numeric ids)
        self.train_triples, self.train_samples,\
        self.valid_triples, self.valid_samples, \
        self.test_triples, self.test_samples = None, None, None, None, None, None

        # Map each (head_id, rel_id) pair to the tail ids that complete the pair in train, valid or test samples.
        # This is used when computing ranks in filtered scenario.
        self.to_filter = defaultdict(lambda: list())

        # Map each (head_id, rel_id) pair to the tail ids that complete the pair in train samples only.
        # This is used by Loss functions that perform negative sampling.
        self.train_to_filter = defaultdict(lambda: list())

        # map each entity and relation id to its training degree
        self.entity_2_degree = defaultdict(lambda: 0)
        self.relation_2_degree = defaultdict(lambda: 0)

        # number of distinct entities and relations in this dataset.
        # Num_relations counts each relation twice, because the dataset adds an inverse fact for each direct one.
        # As a consequence, num_relations = 2*num_direct_relations
        self.num_entities, self.num_relations, self.num_direct_relations = -1, -1, -1

        if load:
            if not os.path.isdir(self.home):
                raise Exception("Folder %s does not exist" % self.home)

            # internal counter for the distinct entities and relations encountered so far
            self._entity_counter, self._relation_counter = 0, 0

            # read train, valid and test triples, and extract the corresponding samples; both are numpy arrays.
            # Triples feature entity and relation names; samples feature the corresponding ids.
            self.train_triples, self.train_samples = self._read_triples(self.train_path, self.separator)
            self.valid_triples, self.valid_samples = self._read_triples(self.valid_path, self.separator)
            self.test_triples, self.test_samples = self._read_triples(self.test_path, self.separator)

            # this is used for O(1) access to training samples
            self.train_samples_set = {(h, r, t) for (h, r, t) in self.train_samples}

            # update the overall number of distinct entities and distinct relations in the dataset
            self.num_entities = len(self.entities)
            self.num_direct_relations = len(self.relations)
            self.num_relations = 2*len(self.relations)  # also count inverse relations

            # add the inverse relations to the relation_id_2_name and relation_name_2_id data structures
            for relation_id in range(self.num_direct_relations):
                inverse_relation_id = relation_id + self.num_direct_relations
                inverse_relation_name = "INVERSE_" + self.relation_id_2_name[relation_id]
                self.relation_id_2_name[inverse_relation_id] = inverse_relation_name
                self.relation_name_2_id[inverse_relation_name] = inverse_relation_id

            # add the tail_id to the list of all tails seen completing (head_id, relation_id, ?)
            # and add the head_id to the list of all heads seen completing (?, relation_id, tail_id)
            all_samples = numpy.vstack((self.train_samples, self.valid_samples, self.test_samples))
            for i in range(all_samples.shape[0]):
                (head_id, relation_id, tail_id) = all_samples[i]
                self.to_filter[(head_id, relation_id)].append(tail_id)
                self.to_filter[(tail_id, relation_id + self.num_direct_relations)].append(head_id)
                # if the sample was a training sample, also do the same for the train_to_filter data structure;
                # Also fill the entity_2_degree and relation_2_degree dicts.
                if i < len(self.train_samples):
                    self.train_to_filter[(head_id, relation_id)].append(tail_id)
                    self.train_to_filter[(tail_id, relation_id + self.num_direct_relations)].append(head_id)
                    self.entity_2_degree[head_id] += 1
                    self.relation_2_degree[relation_id] += 1
                    if tail_id != head_id:
                        self.entity_2_degree[tail_id] += 1

            # map each relation id to its type (ONE_TO_ONE, ONE_TO_MANY, MANY_TO_ONE, or MANY_TO_MANY)
            self._compute_relation_2_type()

    def _read_triples(self, triples_path: str, separator="\t"):
        """
            Private method to read the triples (that is, facts and samples) from a textual file
            :param triples_path: the path of the file to read the triples from
            :param separator: the separator used in the file to read to separate head, relation and tail of each triple
            :return: a 2D numpy array containing the read facts,
                     and a 2D numpy array containing the corresponding samples
        """
        textual_triples = []
        data_triples = []

        with open(triples_path, "r") as triples_file:
            lines = triples_file.readlines()
            for line in lines:
                line = html.unescape(line)      # this is required for some YAGO3-10 lines
                head_name, relation_name, tail_name = line.strip().split(separator)
                textual_triples.append((head_name, relation_name, tail_name))

                self.entities.add(head_name)
                self.entities.add(tail_name)
                self.relations.add(relation_name)

                if head_name in self.entity_name_2_id:
                    head_id = self.entity_name_2_id[head_name]
                else:
                    head_id = self._entity_counter
                    self._entity_counter += 1
                    self.entity_name_2_id[head_name] = head_id
                    self.entity_id_2_name[head_id] = head_name

                if relation_name in self.relation_name_2_id:
                    relation_id = self.relation_name_2_id[relation_name]
                else:
                    relation_id = self._relation_counter
                    self._relation_counter += 1
                    self.relation_name_2_id[relation_name] = relation_id
                    self.relation_id_2_name[relation_id] = relation_name

                if tail_name in self.entity_name_2_id:
                    tail_id = self.entity_name_2_id[tail_name]
                else:
                    tail_id = self._entity_counter
                    self._entity_counter += 1
                    self.entity_name_2_id[tail_name] = tail_id
                    self.entity_id_2_name[tail_id] = tail_name

                data_triples.append((head_id, relation_id, tail_id))

        return numpy.array(textual_triples), numpy.array(data_triples).astype('int64')

    def invert_samples(self, samples: numpy.array):
        """
            This method computes and returns the inverted version of the passed samples.
            :param samples: the direct samples to invert, in the form of a numpy array
            :return: the corresponding inverse samples, in the form of a numpy array
        """
        output = numpy.copy(samples)

        output[:, 0] = output[:, 2]
        output[:, 2] = samples[:, 0]
        output[:, 1] += self.num_direct_relations

        return output


    def _compute_relation_2_type(self):
        """
            This method computes the type of each relation in the dataset based on the self.train_to_filter structure
            (that must have been already computed and populated).
            The mappings relation - relation type are written in the self.relation_2_type dict.
            :return: None
        """
        if len(self.train_to_filter) == 0:
            raise Exception("The dataset has not been loaded yet, so it is not possible to compute relation types yet.")

        relation_2_heads_nums = defaultdict(lambda: list())
        relation_2_tails_nums = defaultdict(lambda: list())

        for (x, relation) in self.train_to_filter:
            if relation >= self.num_direct_relations:
                relation_2_heads_nums[relation - self.num_direct_relations].append(len(self.to_filter[(x, relation)]))
            else:
                relation_2_tails_nums[relation].append(len(self.to_filter[(x, relation)]))

        self.relation_2_type = {}

        for relation in relation_2_heads_nums:
            average_heads_per_tail = numpy.average(relation_2_heads_nums[relation])
            average_tails_per_head = numpy.average(relation_2_tails_nums[relation])

            if average_heads_per_tail > 1.2 and average_tails_per_head > 1.2:
                self.relation_2_type[relation] = MANY_TO_MANY
            elif average_heads_per_tail > 1.2 and average_tails_per_head <= 1.2:
                self.relation_2_type[relation] = MANY_TO_ONE
            elif average_heads_per_tail <= 1.2 and average_tails_per_head > 1.2:
                self.relation_2_type[relation] = ONE_TO_MANY
            else:
                self.relation_2_type[relation] = ONE_TO_ONE

    def get_id_for_entity_name(self, entity_name: str):
        return self.entity_name_2_id[entity_name]

    def get_name_for_entity_id(self, entity_id: int):
        return self.entity_id_2_name[entity_id]

    def get_id_for_relation_name(self, relation_name: str):
        return self.relation_name_2_id[relation_name]

    def get_name_for_relation_id(self, relation_id: int):
        return self.relation_id_2_name[relation_id]

    def add_training_samples(self, samples_to_add: numpy.array):
        """
            Add some samples to the training samples of this dataset.
            The to_filter and train_to_filter data structures will be updated accordingly
            :param samples_to_add: the list of samples to add, in the form of a numpy array
        """
        self.train_samples = numpy.vstack((self.train_samples, samples_to_add))

        for (head, rel, tail) in samples_to_add:
            self.train_samples_set.add((head, rel, tail))
            self.to_filter[(head, rel)].append(tail)
            self.to_filter[(tail, rel + self.num_direct_relations)].append(head)
            self.train_to_filter[(head, rel)].append(tail)
            self.train_to_filter[(tail, rel + self.num_direct_relations)].append(head)

    def sample_to_fact(self, sample_to_convert: Tuple):
        head_id, rel_id, tail_id = sample_to_convert
        return self.entity_id_2_name[head_id], self.relation_id_2_name[rel_id], self.entity_id_2_name[tail_id]

    def fact_to_sample(self, fact_to_convert: Tuple):
        head_name, rel_name, tail_name = fact_to_convert
        return self.entity_name_2_id[head_name], self.relation_name_2_id[rel_name], self.entity_name_2_id[tail_name]

    def remove_training_samples(self, samples_to_remove: numpy.array):
        """
        This method quietly removes a bunch of samples from the training set of this dataset.
        If asked to remove samples that are not present in the training set, this method does nothing and returns False.
        :param samples_to_remove: the samples to remove as a 2D numpy array
                                  in which each row corresponds to one sample to remove

        :return: a 1D array of boolean values as long as passed array of samples to remove;
                 in each position i, it contains True if the i-th passed sample to remove
                 was actually included in the training set, so it was possible to remove it; False otherwise.
        """

        indices_to_remove = []
        removed_samples = []
        output = []
        for sample_to_remove in samples_to_remove:
            head, rel, tail = sample_to_remove

            if (head, rel, tail) in self.train_samples_set:
                index = numpy.where(numpy.all(self.train_samples == sample_to_remove, axis=1))
                indices_to_remove.append(index[0])
                removed_samples.append(self.train_samples[index[0]])

                self.to_filter[(head, rel)].remove(tail)
                self.to_filter[(tail, rel + self.num_direct_relations)].remove(head)
                self.train_samples_set.remove((head, rel, tail))
                output.append(True)
            else:
                output.append(False)

        self.train_samples = numpy.delete(self.train_samples, indices_to_remove, axis=0)
        return output

    def remove_training_sample(self, sample_to_remove: numpy.array):
        """
        This method quietly removes a sample from the training set of this dataset.
        If asked to remove a sample that is not present in the training set, this method does nothing and returns False.
        :param sample_to_remove: the sample to remove

        :return: True if the passed sample was actually included in the training set, so it was possible to remove it;
                 False otherwise.
        """

        head, rel, tail = sample_to_remove

        if (head, rel, tail) in self.train_samples_set:
            index = numpy.where(numpy.all(self.train_samples == sample_to_remove, axis=1))
            self.train_samples = numpy.delete(self.train_samples, index[0], axis=0)

            self.to_filter[(head, rel)].remove(tail)
            self.to_filter[(tail, rel + self.num_direct_relations)].remove(head)
            self.train_samples_set.remove((head, rel, tail))
            return True
        return False

    @staticmethod
    def replace_entity_in_sample(sample, old_entity: int, new_entity:int, as_numpy=True):
        h, r, t = sample
        if h == old_entity:
            h = new_entity
        if t == old_entity:
            t = new_entity
        return numpy.array([h, r, t]) if as_numpy else (h, r, t)

    @staticmethod
    def replace_entity_in_samples(samples, old_entity: int, new_entity:int, as_numpy=True):
        result = []
        for (h, r, t) in samples:
            if h == old_entity:
                h = new_entity
            if t == old_entity:
                t = new_entity
            result.append((h, r, t))

        return numpy.array(result) if as_numpy else result

def home_folder_for(dataset_name:str):
    dataset_home = os.path.join(DATA_PATH, dataset_name)
    if os.path.isdir(dataset_home):
        return dataset_home
    else:
        raise Exception("Folder %s does not exist" % dataset_home)

class KelpieDataset(Dataset):
    """
        Since Datasets handle the correspondence between textual entities and ids,
        the KelpieDataset has the responsibility to decide the id of the kelpie entity
        and to store the train, valid and test samples specific for the original entity and for the kelpie entity

        A KelpieDataset is never *loaded* from file: it is always generated from a pre-existing, already loaded Dataset.

        Nomenclature used in the KelpieDataset:
            - "original entity": the entity to explain the prediction of in the original Dataset;
            - "clone entity": a "fake" entity post-trained with the same training samples as the original entity
            - "kelpie entity": a "fake" entity post-trained with slightly different training samples from the original entity.
                               (e.g. some training samples have been removed, or added).
    """

    def __init__(self,
                 dataset: Dataset,
                 entity_id: int):

        super(KelpieDataset, self).__init__(name=dataset.name,
                                            separator=dataset.separator,
                                            load=False)

        if dataset.num_entities == -1:
            raise Exception("The Dataset passed to initialize a KelpieDataset must be already loaded")

        # the KelpieDataset is now basically empty (because load=False was used in the super constructor)
        # so we must manually copy (and sometimes update) all the important attributes from the original loaded Dataset
        self.num_entities = dataset.num_entities + 1                # adding the Kelpie entity to the count
        self.num_relations = dataset.num_relations
        self.num_direct_relations = dataset.num_direct_relations

        # copy relevant data structures
        self.to_filter = copy.deepcopy(dataset.to_filter)
        self.train_to_filter = copy.deepcopy(dataset.train_to_filter)
        self.entity_name_2_id = copy.deepcopy(dataset.entity_name_2_id)
        self.entity_id_2_name = copy.deepcopy(dataset.entity_id_2_name)
        self.relation_name_2_id = copy.deepcopy(dataset.relation_name_2_id)
        self.relation_id_2_name = copy.deepcopy(dataset.relation_id_2_name)

        # add the kelpie entity
        self.original_entity_id = entity_id
        self.original_entity_name = self.entity_id_2_name[self.original_entity_id]
        self.kelpie_entity_id = dataset.num_entities  # add the kelpie entity to the dataset; it is always the last one
        self.kelpie_entity_name = "kelpie_" + self.original_entity_name
        self.entity_name_2_id[self.kelpie_entity_name] = self.kelpie_entity_id
        self.entity_id_2_name[self.kelpie_entity_id] = self.kelpie_entity_name

        # We do not copy all the triples and samples from the original dataset: the KelpieDataset DOES NOT NEED THEM.
        # The train, valid, and test samples of the KelpieDataset are generated using only those that featured the original entity!
        self.original_train_samples = self._extract_samples_with_entity(dataset.train_samples, self.original_entity_id)
        self.original_valid_samples = self._extract_samples_with_entity(dataset.valid_samples, self.original_entity_id)
        self.original_test_samples = self._extract_samples_with_entity(dataset.test_samples, self.original_entity_id)

        self.kelpie_train_samples = Dataset.replace_entity_in_samples(self.original_train_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_valid_samples = Dataset.replace_entity_in_samples(self.original_valid_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_test_samples = Dataset.replace_entity_in_samples(self.original_test_samples, self.original_entity_id, self.kelpie_entity_id)

        samples_to_stack = [self.kelpie_train_samples]
        if len(self.kelpie_valid_samples) > 0:
            samples_to_stack.append(self.kelpie_valid_samples)
        if len(self.kelpie_test_samples) > 0:
            samples_to_stack.append(self.kelpie_test_samples)
        all_kelpie_samples = numpy.vstack(samples_to_stack)

        for i in range(all_kelpie_samples.shape[0]):
            (head_id, relation_id, tail_id) = all_kelpie_samples[i]
            self.to_filter[(head_id, relation_id)].append(tail_id)
            self.to_filter[(tail_id, relation_id + self.num_direct_relations)].append(head_id)
            # if the sample was a training sample, also do the same for the train_to_filter data structure;
            # Also fill the entity_2_degree and relation_2_degree dicts.
            if i < len(self.kelpie_train_samples):
                self.train_to_filter[(head_id, relation_id)].append(tail_id)
                self.train_to_filter[(tail_id, relation_id + self.num_direct_relations)].append(head_id)

        #for (x, y) in self.train_to_filter:
        #    if self.kelpie_entity_id not in self.train_to_filter[(x, y)]:
        #        self.train_to_filter[(x, y)].append(self.kelpie_entity_id)

        # initialize data structures needed in the case of additions, to allow undoing additions
        self.last_added_samples = []
        self.last_added_samples_number = 0
        self.last_filter_additions = defaultdict(lambda:[])
        self.last_added_kelpie_samples = []


    # override
    def add_training_samples(self, samples_to_add: numpy.array):
        """
            Add an array of training samples to the kelpie training samples of this KelpieDataset.
            The samples to add must still feature the original entity id; this method will convert them before addition.
            The KelpieDataset will keep track of the last performed addition so it can be undone if necessary.

            :param samples_to_add: the samples to add, still featuring the id of the original entity,
                                   in the form of a numpy array
        """

        for sample in samples_to_add:
            assert self.original_entity_id == sample[0] or self.original_entity_id == sample[2]

        self.last_added_samples = samples_to_add
        self.last_added_samples_number = len(samples_to_add)

        # reset all data structures needed to undo additions. We only want to keep track of the *last* addition.
        self.last_filter_additions = defaultdict(lambda:[])
        self.last_added_kelpie_samples = []

        kelpie_samples_to_add = Dataset.replace_entity_in_samples(samples_to_add,
                                                                  old_entity=self.original_entity_id,
                                                                  new_entity=self.kelpie_entity_id)
        for (cur_head, cur_rel, cur_tail) in kelpie_samples_to_add:
            self.to_filter[(cur_head, cur_rel)].append(cur_tail)
            self.to_filter[(cur_tail, cur_rel + self.num_direct_relations)].append(cur_head)
            self.train_to_filter[(cur_head, cur_rel)].append(cur_tail)
            self.train_to_filter[(cur_tail, cur_rel + self.num_direct_relations)].append(cur_head)

            self.last_added_kelpie_samples.append((cur_head, cur_rel, cur_tail))
            self.last_filter_additions[(cur_head, cur_rel)].append(cur_tail)
            self.last_filter_additions[(cur_tail, cur_rel + self.num_direct_relations)].append(cur_head)

        self.kelpie_train_samples = numpy.vstack((self.kelpie_train_samples, numpy.array(kelpie_samples_to_add)))


    def undo_last_training_samples_addition(self):

        # todo: add documentation. the samples to remove are all assumed to feature the kelpie entity
        if self.last_added_samples_number <= 0:
            raise Exception("No addition to undo.")

        self.kelpie_train_samples = self.kelpie_train_samples[:-self.last_added_samples_number]
        for key in self.last_filter_additions:
            for x in self.last_filter_additions[key]:
                self.to_filter[key].remove(x)

        # reset the data structures used to undo additions
        self.last_added_samples = []
        self.last_added_samples_number = 0
        self.last_filter_additions = defaultdict(lambda:[])
        self.last_added_kelpie_samples = []

    def as_kelpie_sample(self, original_sample):
        if not self.original_entity_id in original_sample:
            raise Exception("Could not find the original entity " + str(self.original_entity_id) + " in the passed sample " + str(original_sample))
        return Dataset.replace_entity_in_sample(sample=original_sample,
                                                old_entity=self.original_entity_id,
                                                new_entity=self.kelpie_entity_id)

    def as_original_sample(self, kelpie_sample):
        if not self.kelpie_entity_id in kelpie_sample:
            raise Exception(
                "Could not find the original entity " + str(self.original_entity_id) + " in the passed sample " + str(kelpie_sample))
        return Dataset.replace_entity_in_sample(sample=kelpie_sample,
                                                old_entity=self.kelpie_entity_id,
                                                new_entity=self.original_entity_id)

    ### private utility methods
    @staticmethod
    def _extract_samples_with_entity(samples, entity_id):
        return samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]
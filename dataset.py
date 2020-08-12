import copy
import html
import os
from collections import defaultdict

import numpy

from config import ROOT

DATA_PATH = os.path.join(ROOT, "data")
FB15K = "FB15k"
FB15K_237 = "FB15k-237"
WN18 = "WN18"
WN18RR = "WN18RR"
YAGO3_10 = "YAGO3-10"

ALL_DATASET_NAMES = [FB15K, FB15K_237, WN18, WN18RR, YAGO3_10]

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

        # map that associates to each couple (head, relation) all the tails that have been seen completing them
        # either in train samples, or in valid samples, or in test samples
        self.to_filter = defaultdict(lambda: list())

        # number of distinct entities and relations in this dataset.
        # num_direct_relations is used by extensions of Dataset that may support inverse relations as well
        self.num_entities, self.num_relations, self.num_direct_relations = -1, -1, -1

        if load:
            if not os.path.isdir(self.home):
                raise Exception("Folder %s does not exist" % self.home)

            # internal counter for the distinct entities and relations encountered so far
            self._entity_counter, self._relation_counter = 0, 0

            # read train, valid and test set triples with entities and relations
            # both in their textual format (train_triples, valid_triples and test_triples lists)
            # and in their numeric format (train_data, valid_data and test_data )
            # all these collections are numpy arrays
            self.train_triples, self.train_samples = self._read_triples(self.train_path, self.separator)
            self.valid_triples, self.valid_samples = self._read_triples(self.valid_path, self.separator)
            self.test_triples, self.test_samples = self._read_triples(self.test_path, self.separator)

            # update the overall number of distinct entities and distinct relations in the dataset
            self.num_entities = len(self.entities)
            self.num_direct_relations = len(self.relations)
            self.num_relations = 2*len(self.relations)  # also include inverse ones

            # add the inverse relations to the relation_id_2_name and relation_name_2_id data structures
            for relation_id in range(self.num_direct_relations):
                inverse_relation_id = relation_id + self.num_direct_relations
                inverse_relation_name = "INVERSE_" + self.relation_id_2_name[relation_id]
                self.relation_id_2_name[inverse_relation_id] = inverse_relation_name
                self.relation_name_2_id[inverse_relation_name] = inverse_relation_id

            # add the tail_id to the list of all tails seen completing (head_id, relation_id, ?)
            # and add the head_id to the list of all heads seen completing (?, relation_id, tail_id)
            for sample in numpy.vstack((self.train_samples, self.valid_samples, self.test_samples)):
                (head_id, relation_id, tail_id) = sample
                self.to_filter[(head_id, relation_id)].append(tail_id)
                self.to_filter[(tail_id, relation_id + self.num_direct_relations)].append(head_id)


    def _read_triples(self, triples_path: str, separator="\t"):
        """
        Private method to read the triples from a textual file,
        and to use the obtained triples to progressively fill the dataset data structures in the meantime.

        :param triples_path:
        :param separator:
        :return:
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
        output = numpy.copy(samples)

        tmp = numpy.copy(output[:, 0])
        output[:, 0] = output[:, 2]
        output[:, 2] = tmp
        output[:, 1] += self.num_direct_relations

        return output


    def get_id_for_entity_name(self, entity_name: str):
        return self.entity_name_2_id[entity_name]

    def get_name_for_entity_id(self, entity_id: int):
        return self.entity_id_2_name[entity_id]

    def get_id_for_relation_name(self, relation_name: str):
        return self.relation_name_2_id[relation_name]

    def get_name_for_relation_id(self, relation_id: int):
        return self.relation_id_2_name[relation_id]


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
    """

    def __init__(self,
                 dataset: Dataset,
                 entity_id: int):

        super(KelpieDataset, self).__init__(name=dataset.name,
                                            separator=dataset.separator,
                                            load=False)

        if dataset.num_entities == -1:
            raise Exception("The ComplExDataset passed to initialize a KelpieComplExDataset must be already loaded")

        # the KelpieDataset is now basically empty (because load=False was used in the super constructor)
        # so we must manually copy (and sometimes update) all the important attributes from the loaded Dataset
        self.num_entities = dataset.num_entities + 1    # add the Kelpie entity to the count!
        self.num_relations = dataset.num_relations
        self.num_direct_relations = dataset.num_direct_relations

        self.to_filter = copy.deepcopy(dataset.to_filter)
        self.entity_name_2_id = copy.deepcopy(dataset.entity_name_2_id)
        self.entity_id_2_name = copy.deepcopy(dataset.entity_id_2_name)
        self.relation_name_2_id = copy.deepcopy(dataset.relation_name_2_id)
        self.relation_id_2_name = copy.deepcopy(dataset.relation_id_2_name)

        # in our nomenclature the "original entity" is the entity to explain the prediction of;
        # the "kelpie entity "is the fake entity created (and later post-trained) to explain the original entity
        self.original_entity_id = entity_id
        self.original_entity_name = self.entity_id_2_name[self.original_entity_id]
        # add the kelpie entity to the dataset
        self.kelpie_entity_id = dataset.num_entities       # the new entity is always the last one

        self.entity_name_2_id["kelpie"] = self.kelpie_entity_id
        self.entity_id_2_name[self.kelpie_entity_id] = "kelpie"

        # note that we are not copying all the triples and samples from the original dataset,
        # because the KelpieComplExDataset DOES NOT NEED THEM.
        # The train, valid, and test samples of the KelpieComplExDataset
        # are only those of the original dataset that feature the original entity!
        self.original_train_samples = self._extract_samples_with_entity(dataset.train_samples, self.original_entity_id)
        self.original_valid_samples = self._extract_samples_with_entity(dataset.valid_samples, self.original_entity_id)
        self.original_test_samples = self._extract_samples_with_entity(dataset.test_samples, self.original_entity_id)

        # build the train, valid and test samples (both direct and inverse) for the kelpie entity
        # by replacing the original entity id with the kelpie entity id
        self.kelpie_train_samples = self._replace_entity_in_samples(self.original_train_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_valid_samples = self._replace_entity_in_samples(self.original_valid_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_test_samples = self._replace_entity_in_samples(self.original_test_samples, self.original_entity_id, self.kelpie_entity_id)

        # update the to_filter sets to include the filter lists for the facts with the Kelpie entity
        for (entity_id, relation_id) in dataset.to_filter:
            if entity_id == self.original_entity_id:
                self.to_filter[(self.kelpie_entity_id, relation_id)] = copy.deepcopy(self.to_filter[(entity_id, relation_id)])

        # add the kelpie entity in the filter list for all original facts
        for (entity_id, relation_id) in self.to_filter:
            # if the couple (entity_id, relation_id) was in the original dataset,
            # ALWAYS add the kelpie entity to the filtering list
            if (entity_id, relation_id) in dataset.to_filter:
                self.to_filter[(entity_id, relation_id)].append(self.kelpie_entity_id)

            # else, it means that the entity id is the kelpie entity id.
            # in this case add the kelpie entity id to the list only if the original entity id is already in the list
            elif self.original_entity_id in self.to_filter[(entity_id, relation_id)]:
                self.to_filter[(entity_id, relation_id)].append(self.kelpie_entity_id)

    ### private utility methods
    @staticmethod
    def _extract_samples_with_entity(samples, entity_id):
        return samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]

    @staticmethod
    def _replace_entity_in_samples(samples, old_entity_id, new_entity_id):
        result = numpy.copy(samples)
        for i in range(len(result)):
            if result[i, 0] == old_entity_id:
                result[i, 0] = new_entity_id
            if result[i, 2] == old_entity_id:
                result[i, 2] = new_entity_id

        return result
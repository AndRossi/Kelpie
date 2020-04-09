import html
import os
from collections import defaultdict

DATA_PATH = os.path.abspath("/Users/andrea/kelpie/data")
FB15K = "FB15k"
FB15K_237 = "FB15k-237"
WN18 = "WN18"
WN18RR = "WN18RR"
YAGO3_10 = "YAGO3-10"

ALL_DATASET_NAMES = [FB15K, FB15K_237, WN18, WN18RR, YAGO3_10]

class Dataset:

    def __init__(self, name, separator="\t"):
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
        self.entity_name_2_id, self.entity_id_2_name = defaultdict(lambda: None), defaultdict(lambda: None)
        self.relation_name_2_id, self.relation_id_2_name = defaultdict(lambda: None), defaultdict(lambda: None)

        # collections of triples (facts with textual names) and samples (facts with numeric ids)
        self.train_triples, self.train_samples,\
        self.valid_triples, self.valid_samples, \
        self.test_triples, self.test_samples = None, None, None, None, None, None

        # number of distinct entities and relations in this dataset
        self.num_entities, self.num_relations = -1, -1

    def load(self):
        if not os.path.isdir(self.home):
            raise Exception("Folder %s does not exist" % self.home)

        # internal counter for the distinct entities and relations encountered so far
        self._entity_counter, self._relation_counter = 0, 0

        # read train, valid and test set triples with entities and relations
        # both in their textual format (train_triples, valid_triples and test_triples lists)
        # and in their numeric format (train_data, valid_data and test_data )
        self.train_triples, self.train_samples = self._read_triples(self.train_path, self.separator)
        self.valid_triples, self.valid_samples = self._read_triples(self.valid_path, self.separator)
        self.test_triples, self.test_samples = self._read_triples(self.test_path, self.separator)

        # update the overall number of distinct entities and distinct relations in the dataset
        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)

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

                head_id = self.entity_name_2_id[head_name]
                if head_id is None:
                    head_id = self._entity_counter
                    self._entity_counter += 1
                    self.entity_name_2_id[head_name] = head_id
                    self.entity_id_2_name[head_id] = head_name

                relation_id = self.relation_name_2_id[relation_name]
                if relation_id is None:
                    relation_id = self._relation_counter
                    self._relation_counter += 1
                    self.relation_name_2_id[relation_name] = relation_id
                    self.relation_id_2_name[relation_id] = relation_name

                tail_id = self.entity_name_2_id[tail_name]
                if tail_id is None:
                    tail_id = self._entity_counter
                    self._entity_counter += 1
                    self.entity_name_2_id[tail_name] = tail_id
                    self.entity_id_2_name[tail_id] = tail_name

                data_triples.append((head_id, relation_id, tail_id))

        return textual_triples, data_triples


def home_folder_for(dataset_name):
    dataset_home = os.path.join(DATA_PATH, dataset_name)
    if os.path.isdir(dataset_home):
        return dataset_home
    else:
        raise Exception("Folder %s does not exist" % dataset_home)

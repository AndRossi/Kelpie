import copy
from collections import defaultdict
import numpy as np
from kelpie.dataset import Dataset, FB15K


class ComplExDataset(Dataset):
    """
        Dataset usable by ComplEx
        It must be a different class from the general Dataset because it handles inverse relations in a different way..
        So, this ComplExDataset object has the responsibility to "know" that inverse relations are handled...
    """
    def __init__(self,
                 name: str,
                 separator = "\t"):

        super(ComplExDataset, self).__init__(name, separator)

        # the number of original relations is different from the number of actual relations,
        # because in a ComplEx dataset each relation spawns two distinct relations: the direct and the inverse one
        self.num_original_relations = -1
        self.num_relations = -1

        # build the sets of entities to filter away
        # when performing head prediction or tail prediction in filtered scenario:
        # each couple (entity, relation) is mapped to the set of "correct answers" seen in training, valid or test set
        self.to_filter = defaultdict(list)

        # the ComplEx dataset has to also store inverse versions of all triples
        self.inverse_train_samples, self.inverse_valid_samples, self.inverse_test_samples = None, None, None


    def load(self):

        # load the dataset as you would load any dataset
        super().load()

        # set the num_original_relations and num_relations according to the values updated with load
        self.num_original_relations = self.num_relations
        self.num_relations *= 2

        # populate the filtering lists for both head and tail predictions
        for (head_id, relation_id, tail_id) in self.train_samples + self.valid_samples + self.test_samples:
            self.to_filter[(head_id, relation_id)].append(tail_id)
            self.to_filter[(tail_id, relation_id + self.num_original_relations)].append(head_id)

        # converting all lists of samples to numpy arrays
        self.train_samples = np.array(self.train_samples).astype('int64')
        self.valid_samples = np.array(self.valid_samples).astype('int64')
        self.test_samples = np.array(self.test_samples).astype('int64')

        # redoubling the samples sets set in order to include the "inverse" facts
        # obtained by switching heads with tails, and adding "num_original_relations" to the id of the relation
        self.inverse_train_samples = self.invert_samples(self.train_samples)
        self.inverse_valid_samples = self.invert_samples(self.valid_samples)
        self.inverse_test_samples = self.invert_samples(self.test_samples)

    # whenever you get the samples from this dataset, you can either request the "stacked" version
    # (in which direct and inverse facts are in the same numpy array) ---> this is good in training
    # or the "non stacked version" in which they are separete numpy arrays ---> this is good in testing

    def get_train_samples(self, with_inverse=False):
        return np.vstack((self.train_samples, self.inverse_train_samples)) if with_inverse else self.train_samples

    def get_valid_samples(self, with_inverse=False):
        return np.vstack((self.valid_samples, self.inverse_valid_samples)) if with_inverse else self.valid_samples

    def get_test_samples(self, with_inverse=False):
        return np.vstack((self.test_samples, self.inverse_test_samples)) if with_inverse else self.test_samples


    def invert_samples(self, samples: np.array):
        output = np.copy(samples)

        tmp = np.copy(output[:, 0])
        output[:, 0] = output[:, 2]
        output[:, 2] = tmp
        output[:, 1] += self.num_original_relations

        return output

    def get_id_for_entity_name(self, entity_name):
        return self.entity_name_2_id[entity_name]

    def get_name_for_entity_id(self, entity_id):
        return self.entity_id_2_name[entity_id]

    def get_id_for_relation_name(self, relation_name):
        return self.relation_name_2_id[relation_name]

    def get_name_for_relation_id(self, relation_id):
        return self.relation_id_2_name[relation_id]


class KelpieComplExDataset(ComplExDataset):
    """
        Since Datasets handle the correspondence between textual entities and ids,
        this KelpieComplExDataset also has the responsibility to decide the id of the kelpie entity
        and to store the train, valid and test samples specific for the original entity and for the kelpie entity
    """


    # this dataset is weird because it is never *loaded*,
    # but is always generated from a pre-existing ComplExDataset!
    def __init__(self,
                 dataset: ComplExDataset,
                 entity_id: int):

        super(KelpieComplExDataset, self).__init__(dataset.name, dataset.separator)

        if dataset.num_entities == -1:
            raise Exception("The ComplExDataset passed to initialize a KelpieComplExDataset must be already loaded")

        # in our nomenclature the "original entity" is the entity to explain the prediction of;
        # the "kelpie entity "is the fake entity created (and later post-trained) to explain the original entity
        self.original_entity_id = entity_id
        self.original_entity_name = self.get_name_for_entity_id(self.original_entity_id)
        # add the kelpie entity to the dataset
        self.kelpie_entity_id = dataset.num_entities       # the new entity is always the last one
        self.entity_name_2_id["kelpie"] = self.kelpie_entity_id
        self.entity_id_2_name[self.kelpie_entity_id] = "kelpie"

        # copy and update, if necessary, some attributes from the pre-existing ComplExDataset
        self.num_entities = dataset.num_entities + 1    # add the Kelpie entity to the count!
        self.num_relations = dataset.num_relations
        self.num_original_relations = dataset.num_original_relations
        self.to_filter = copy.deepcopy(dataset.to_filter)

        # note that we are not copying the triples and samples from the original dataset,
        # because the KelpieComplExDataset DOES NOT NEED THEM.
        # The train, valid, and test samples (and their inverse counterparts) of the KelpieComplExDataset
        # are only those of the original dataset that feature the original entity!
        self.train_samples = self._extract_samples_with_entity(dataset.train_samples, self.original_entity_id)
        self.valid_samples = self._extract_samples_with_entity(dataset.valid_samples, self.original_entity_id)
        self.test_samples = self._extract_samples_with_entity(dataset.test_samples, self.original_entity_id)
        self.inverse_train_samples = self.invert_samples(self.train_samples)
        self.inverse_valid_samples = self.invert_samples(self.valid_samples)
        self.inverse_test_samples = self.invert_samples(self.test_samples)

        # build the train, valid and test samples (both direct and inverse) for the kelpie entity
        # by replacing the original entity id with the kelpie entity id
        self.kelpie_train_samples = self._replace_entity_in_samples(self.train_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_valid_samples = self._replace_entity_in_samples(self.valid_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_test_samples = self._replace_entity_in_samples(self.test_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_inverse_train_samples = self._replace_entity_in_samples(self.inverse_train_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_inverse_valid_samples = self._replace_entity_in_samples(self.inverse_valid_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_inverse_test_samples = self._replace_entity_in_samples(self.inverse_test_samples, self.original_entity_id, self.kelpie_entity_id)

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


    #override
    def load(self):
        raise Exception("A KelpieComplExDataset can not be loaded: it is built from a pre-existing loaded ComplExDataset")


    def get_kelpie_train_samples(self, with_inverse=False):
        return np.vstack((self.kelpie_train_samples, self.kelpie_inverse_train_samples)) if with_inverse \
            else self.kelpie_train_samples

    def get_kelpie_valid_samples(self, with_inverse=False):
        return np.vstack((self.kelpie_valid_samples, self.kelpie_inverse_valid_samples)) if with_inverse \
            else self.kelpie_valid_samples

    def get_kelpie_test_samples(self, with_inverse=False):
        return np.vstack((self.kelpie_test_samples, self.kelpie_inverse_test_samples)) if with_inverse \
            else self.kelpie_test_samples

    ### private utility methods

    def _extract_samples_with_entity(self, samples, entity_id):
        return samples[np.where(np.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]

    def _replace_entity_in_samples(self, samples, old_entity_id, new_entity_id):
        result = np.copy(samples)
        result[result == old_entity_id] = new_entity_id
        return result
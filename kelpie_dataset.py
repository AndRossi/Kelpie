import copy
from collections import defaultdict
import numpy
from dataset import Dataset

class KelpieDataset(Dataset):
    """
        Since Datasets handle the correspondence between textual entities and ids,
        the KelpieDataset has the responsibility to decide the id of the kelpie entity (aka mimic in our paper)
        and to store the train, valid and test samples specific for the original entity and for the kelpie entity

        A KelpieDataset is never *loaded* from file: it is always generated from a pre-existing, already loaded Dataset.

        Nomenclature used in the KelpieDataset:
            - "original entity": the entity to explain the prediction of in the original Dataset;
            - "clone entity": a homologous mimic, i.e., a "fake" entity
                              post-trained with the same training samples as the original entity
            - "kelpie entity": a non-homologous mimic, i.e., a "fake" entity
                               post-trained with slightly different training samples from the original entity.
                               (e.g. some training samples may have been removed, or added).
    """

    def __init__(self,
                 dataset: Dataset,
                 entity_ids):  # 一般不指定tail，除非需要寻找relation_path

        super(KelpieDataset, self).__init__(name=dataset.name,
                                            separator=dataset.separator,
                                            load=False,
                                            args=dataset.args)
        if dataset.num_entities == -1:
            raise Exception("The Dataset passed to initialize a KelpieDataset must be already loaded")

        # the KelpieDataset is now basically empty (because load=False was used in the super constructor)
        # so we must manually copy (and sometimes update) all the important attributes from the original loaded Dataset
        if hasattr(entity_ids, '__iter__'):
            self.entity_ids = entity_ids
        else:
            self.entity_ids = [entity_ids]
        
        print('\tkelpie entity_ids:', self.entity_ids)
        self.l = len(self.entity_ids)
        self.num_entities = dataset.num_entities + self.l                # adding the Kelpie entity to the count
        self.num_relations = dataset.num_relations
        self.num_direct_relations = dataset.num_direct_relations
        self.dataset = dataset

        # copy relevant data structures
        self.to_filter = copy.deepcopy(dataset.to_filter)
        self.train_to_filter = copy.deepcopy(dataset.train_to_filter)
        self.entity_name_2_id = copy.deepcopy(dataset.entity_name_2_id)
        self.entity_id_2_name = copy.deepcopy(dataset.entity_id_2_name)
        self.relation_name_2_id = copy.deepcopy(dataset.relation_name_2_id)
        self.relation_id_2_name = copy.deepcopy(dataset.relation_id_2_name)
        self.rid2target = copy.deepcopy(dataset.rid2target)
        if dataset.tail_restrain:
            self.tail_restrain = copy.deepcopy(dataset.tail_restrain)
        if dataset.g:
            # https://stackoverflow.com/questions/60310123/why-is-graph-copy-slower-than-copy-deepcopygraph-in-networkx
            self.g = copy.deepcopy(dataset.g)

        # add the kelpie entity
        self.kelpie_ids = []     # 最后l个分别映射到entity_ids
        self.original_train_samples = []
        self.original_valid_samples = []
        self.original_test_samples = []
        self.entity2kelpie = {}

        for ix, entity_id in enumerate(self.entity_ids):
            entity_name = self.entity_id_2_name[entity_id]
            kelpie_name = "kelpie_" + entity_name
            kelpie_id = dataset.num_entities + ix
            self.entity_name_2_id[kelpie_name] = kelpie_id
            self.entity_id_2_name[kelpie_id] = kelpie_name
            self.kelpie_ids.append(kelpie_id)
            self.entity2kelpie[entity_id] = kelpie_id

            # We do not copy all the triples and samples from the original dataset: the KelpieDataset DOES NOT NEED THEM.
            # The train, valid, and test samples of the KelpieDataset are generated using only those that featured the original entity!
            self.original_train_samples.append(self._extract_samples_with_entity(dataset.train_samples, entity_id))
            self.original_valid_samples.append(self._extract_samples_with_entity(dataset.valid_samples, entity_id))
            self.original_test_samples.append(self._extract_samples_with_entity(dataset.test_samples, entity_id))
        self.original_train_samples = numpy.concatenate(self.original_train_samples, axis=0)
        self.original_valid_samples = numpy.concatenate(self.original_valid_samples, axis=0)
        self.original_test_samples = numpy.concatenate(self.original_test_samples, axis=0)

        self.kelpie_train_samples = self.original_train_samples
        self.kelpie_valid_samples = self.original_valid_samples
        self.kelpie_test_samples = self.original_test_samples
        for ix, entity_id in enumerate(self.entity_ids):
            kelpie_id = dataset.num_entities + ix
            self.kelpie_train_samples = Dataset.replace_entity_in_samples(self.kelpie_train_samples, entity_id, kelpie_id)
            self.kelpie_valid_samples = Dataset.replace_entity_in_samples(self.kelpie_valid_samples, entity_id, kelpie_id)
            self.kelpie_test_samples = Dataset.replace_entity_in_samples(self.kelpie_test_samples, entity_id, kelpie_id)

        print(f'\tkelpie dataset created: train {len(self.kelpie_train_samples)}, valid {len(self.kelpie_valid_samples)}, test {len(self.kelpie_test_samples)}')
        # update to_filter and train_to_filter data structures
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

        # create a map that associates each kelpie train_sample to its index in self.kelpie_train_samples
        # this will be necessary to allow efficient removals and undoing removals
        self.kelpie_train_sample_2_index = {}
        for i in range(len(self.kelpie_train_samples)):
            cur_head, cur_rel, cur_tail = self.kelpie_train_samples[i]
            self.kelpie_train_sample_2_index[(cur_head, cur_rel, cur_tail)] = i

        # initialize data structures needed in the case of additions and/or removals;
        # these structures are required to undo additions and/or removals
        self.kelpie_train_samples_copy = copy.deepcopy(self.kelpie_train_samples)

        self.last_added_samples = []
        self.last_added_samples_number = 0
        self.last_filter_additions = defaultdict(lambda:[])
        self.last_added_kelpie_samples = []

        self.last_removed_samples = []
        self.last_removed_samples_number = 0
        self.last_filter_removals = defaultdict(lambda:[])
        self.last_removed_kelpie_samples = []


    # override
    def add_training_samples(self, samples_to_add: numpy.array):
        """
            Add a set of training samples to the training samples of the kelpie entity of this KelpieDataset.
            The samples to add must still feature the original entity id; this method will convert them before addition.
            The KelpieDataset will keep track of the last performed addition so it can be undone if necessary
            calling the undo_last_training_samples_addition method.

            :param samples_to_add: the samples to add, still featuring the id of the original entity,
                                   in the form of a numpy array
        """

        for sample in samples_to_add:
            assert sample[0] in self.entity_ids or sample[2] in self.entity_ids

        self.last_added_samples = samples_to_add
        self.last_added_samples_number = len(samples_to_add)

        # reset all data structures needed to undo additions. We only want to keep track of the *last* addition.
        self.last_filter_additions = defaultdict(lambda:[])
        self.last_added_kelpie_samples = []

        kelpie_samples_to_add = samples_to_add
        for ix, entity_id in enumerate(self.entity_ids):
            kelpie_id = self.dataset.num_entities + ix
            kelpie_samples_to_add = Dataset.replace_entity_in_samples(kelpie_samples_to_add,
                                                                    old_entity=entity_id,
                                                                    new_entity=kelpie_id)

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
        """
            This method undoes the last addition performed on this KelpieDataset
            calling its add_training_samples method.

            The purpose of undoing the additions performed on a pre-existing KelpieDataset,
            instead of creating a new KelpieDataset from scratch, is to improve efficiency.
        """

        if self.last_added_samples_number <= 0:
            raise Exception("No addition to undo.")

        # revert the self.kelpie_train_samples to the self.kelpie_train_samples_copy
        self.kelpie_train_samples = copy.deepcopy(self.kelpie_train_samples_copy)

        # undo additions to to_filter and train_to_filter
        for key in self.last_filter_additions:
            for x in self.last_filter_additions[key]:
                self.to_filter[key].remove(x)
                self.train_to_filter[key].remove(x)

        # reset the data structures used to undo additions
        self.last_added_samples = []
        self.last_added_samples_number = 0
        self.last_filter_additions = defaultdict(lambda:[])
        self.last_added_kelpie_samples = []

    def assert_samples_in(self, sample, entity_ids=None):
        if entity_ids is None:
            entity_ids = self.entity_ids
        assert sample[0] in entity_ids or sample[2] in entity_ids
        # if not self.head_id in original_sample:
        #     raise Exception("Could not find the original entity " + str(self.head_id) + " in the passed sample " + str(original_sample))s

    # override
    def remove_training_samples(self, samples_to_remove: numpy.array):
        """
            Remove some training samples from the kelpie training samples of this KelpieDataset.
            The samples to remove must still feature the original entity id; this method will convert them before removal.
            The KelpieDataset will keep track of the last performed removal so it can be undone if necessary.

            :param samples_to_remove: the samples to add, still featuring the id of the original entity,
                                   in the form of a numpy array
        """

        for sample in samples_to_remove:
            self.assert_samples_in(sample)

        self.last_removed_samples = samples_to_remove
        self.last_removed_samples_number = len(samples_to_remove)

        # reset data structures needed to undo removals. We only want to keep track of the *last* removal.
        self.last_filter_removals = defaultdict(lambda:[])
        self.last_removed_kelpie_samples = []

        kelpie_train_samples_to_remove = samples_to_remove
        for ix, entity_id in enumerate(self.entity_ids):
            kelpie_id = self.dataset.num_entities + ix
            kelpie_train_samples_to_remove = Dataset.replace_entity_in_samples(samples=kelpie_train_samples_to_remove,
                                                                           old_entity=entity_id,
                                                                           new_entity=kelpie_id,
                                                                           as_numpy=False)

        # update to_filter and train_to_filter
        for (cur_head, cur_rel, cur_tail) in kelpie_train_samples_to_remove:
            self.to_filter[(cur_head, cur_rel)].remove(cur_tail)
            self.to_filter[(cur_tail, cur_rel + self.num_direct_relations)].remove(cur_head)
            self.train_to_filter[(cur_head, cur_rel)].remove(cur_tail)
            self.train_to_filter[(cur_tail, cur_rel + self.num_direct_relations)].remove(cur_head)

            # and also update the data structures required for undoing the removal
            self.last_removed_kelpie_samples.append((cur_head, cur_rel, cur_tail))
            self.last_filter_removals[(cur_head, cur_rel)].append(cur_tail)
            self.last_filter_removals[(cur_tail, cur_rel + self.num_direct_relations)].append(cur_head)

        # get the indices of the samples to remove in the kelpie_train_samples structure
        # and use them to perform the actual removal
        kelpie_train_indices_to_remove = [self.kelpie_train_sample_2_index[x] for x in kelpie_train_samples_to_remove]
        self.kelpie_train_samples = numpy.delete(self.kelpie_train_samples, kelpie_train_indices_to_remove, axis=0)


    def undo_last_training_samples_removal(self):
        """
            This method undoes the last removal performed on this KelpieDataset
            calling its add_training_samples method.

            The purpose of undoing the removals performed on a pre-existing KelpieDataset,
            instead of creating a new KelpieDataset from scratch, is to improve efficiency.
        """
        if self.last_removed_samples_number <= 0:
            raise Exception("No removal to undo.")

        # revert the self.kelpie_train_samples to the self.kelpie_train_samples_copy
        self.kelpie_train_samples = copy.deepcopy(self.kelpie_train_samples_copy)

        # undo additions to to_filter and train_to_filter
        for key in self.last_filter_removals:
            for x in self.last_filter_removals[key]:
                self.to_filter[key].append(x)
                self.train_to_filter[key].append(x)

        # reset the data structures used to undo additions
        self.last_removed_samples = []
        self.last_removed_samples_number = 0
        self.last_filter_removals = defaultdict(lambda:[])
        self.last_removed_kelpie_samples = []


    def as_kelpie_sample(self, original_sample):
        self.assert_samples_in(original_sample)
        
        for ix, entity_id in enumerate(self.entity_ids):
            kelpie_id = self.dataset.num_entities + ix
            original_sample = Dataset.replace_entity_in_sample(sample=original_sample,
                                                old_entity=entity_id,
                                                new_entity=kelpie_id)
        return original_sample


    def as_original_sample(self, kelpie_sample):
        self.assert_samples_in(kelpie_sample, self.kelpie_ids)
        
        for ix, entity_id in enumerate(self.entity_ids):
            kelpie_id = self.dataset.num_entities + ix
            kelpie_sample = Dataset.replace_entity_in_sample(sample=kelpie_sample,
                                                old_entity=kelpie_id,
                                                new_entity=entity_id)
        return kelpie_sample

    ### private utility methods
    @staticmethod
    def _extract_samples_with_entity(samples, entity_id):
        return samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]
import os
from collections import defaultdict

import numpy
import torch

from dataset import Dataset, MANY_TO_ONE, ONE_TO_ONE
from model import Model

def extract_comparable_entities(model: Model,
                                dataset: Dataset,
                                sample: numpy.array,
                                perspective: str,
                                top_k: int):


    head, relation, tail = sample
    entity_to_explain, target_entity = (head, tail) if perspective == "head" else (tail, head)

    head_rel_couples, rel_tail_couples = _compute_training_couples(dataset)
    # extract the ids of all entities for which
    #   <entity, relation, tail> is not present in the training set if perspective is head
    # or
    #   <head, relation, entity> is not present in the training set if perspective is tail
    other_entities = []
    samples = []

    entity_2_types, type_2_entities = read_entity_types(dataset)

    if perspective == "head":
        for cur_entity in range(0, dataset.num_entities):
            # do not include the entity to explain, of course
            if cur_entity == entity_to_explain:
                continue

            ## if the relation is *_TO_ONE, ignore any entities for which in training, there is already a fact <cur_entity, relation, *>
            if dataset.relation_2_type[relation] in [ONE_TO_ONE, MANY_TO_ONE] and (cur_entity, relation) in head_rel_couples:
                continue

            # if the entity does not have any types in common with the entity to explain, ignore it
            if len(entity_2_types[cur_entity].intersection(entity_2_types[entity_to_explain])) == 0:
                continue

            other_entities.append(cur_entity)
            samples.append((cur_entity, relation, tail))
    else:
        # todo: this is currently not allowed because we would need to collect (head, relation, entity) for all other entities
        #       and, for each (head, relation, entity), we would need to score all possible heads
        #       all_scores currently only supports tail prediction
        #       so basically:
        # todo: add an optional boolean "head_prediction" (default=False); if it is true, compute scores for all heads rather than tails
        raise NotImplementedError
        #for entity in range(0, dataset.num_entities):
            #if entity == entity_to_explain or (head, relation, entity) in dataset.train_samples:
            #    continue
            #other_entities.append(entity)
            #samples.append([head, relation, entity])

    # if no entities satisfy the requirements, return none
    if len(samples) == 0:
        return None

    #entity_to_explain_score = model.score(samples=sample)[0]
    samples_all_scores = model.all_scores(samples=numpy.array(numpy.array(samples))).detach().cpu().numpy()

    other_entity_2_resistance = {}
    for i in range(len(other_entities)):
        cur_other_entity = other_entities[i]
        cur_sample = samples[i]                 # the sample containing the cur_other_entity as head or tail
        cur_head, cur_rel, cur_tail = cur_sample
        cur_sample_all_scores = samples_all_scores[i]

        # WORK IN FILTERED SCENARIO
        # TODO: all of this now only works when perspective is head. Implement for tail too.
        cur_sample_target_score = cur_sample_all_scores[cur_tail]
        # get the list of tails to filter out; this will include the actual target tail entity too
        filter_out = dataset.to_filter[(cur_head, cur_rel)]
        # filter away the score of the "correct" tails (we are in filtered scenario).
        # NOTE: THIS MAY ALSO INCLUDE THE cur_tail
        cur_sample_all_scores[torch.LongTensor(filter_out)] = -1e6
        # So restore  the cur_tail score just in case
        cur_sample_all_scores[cur_tail] = cur_sample_target_score

        cur_sample_best_score = numpy.max(cur_sample_all_scores)

        cur_other_entity_resistance_to_conversion = cur_sample_best_score-cur_sample_target_score

        # if resistance to conversion is 0, current_sample ranked in 1st position the target entity,
        # so the current_other_entity is useless for our purpose
        # (we want current_other_entities that, without additions, do not predict the target entity with rank 1)
        # (e.g. if we are explaining <Barack, nationality, USA>, Xin is an acceptable "convertible entity"
        # only if <Xin, nationality, ?> does not already rank USA in 1st position!)
        if cur_other_entity_resistance_to_conversion <= 0:
            continue
        else:
            other_entity_2_resistance[cur_other_entity] = cur_other_entity_resistance_to_conversion

    # sort by resistance in ascending order
    other_entity_resistance_couples = sorted(other_entity_2_resistance.items(), key=lambda x: x[1], reverse=False)

    return other_entity_resistance_couples[:top_k]


def extract_entities_by_proximity(model: Model,
                                  entity: int,
                                  other_entities: list,
                                  top_k: float):

    all_entity_embeddings = model.entity_embeddings.detach().cpu().numpy()
    entity_embedding = all_entity_embeddings[entity, :]

    entities_with_distances = []
    for other_entity in other_entities:
        other_entity_embedding = all_entity_embeddings[other_entity, :]
        entities_with_distances.append((other_entity, numpy.linalg.norm(entity_embedding-other_entity_embedding, ord=2)))

    entities_with_distances = sorted(entities_with_distances, key=lambda x: x[1], reverse=False)

    # return the first X% couples, where X is the convertibility_param
    return entities_with_distances[:top_k]

def extract_entities_by_adjacent_relations(dataset: Dataset,
                                           entity_id: int,
                                           top_k: float):
    # get the fact to explain and its perspective entity

    entity_2_vector = defaultdict(lambda: numpy.zeros(dataset.num_relations))
    for sample in dataset.train_samples:
        cur_head, cur_relation, cur_tail = sample
        entity_2_vector[cur_head][cur_relation] += 1
        entity_2_vector[cur_tail][cur_relation + dataset.num_direct_relations] += 1

    entity_cossim = []
    for entity in entity_2_vector:
        if entity == entity_id:
            continue
        else:
            similarity = cossim(entity_2_vector[entity_id], entity_2_vector[entity])
            entity_cossim.append((entity, similarity))

    return sorted(entity_cossim, reverse=True, key=lambda x: x[1])[:top_k]


# todo: this can be significantly sped up by making the entity similarity component become a stateful object, that at creation analyzes the training samples and stores the (head, relation) couples in a O(1) data structure
# note: dataset.to_filter is not enough because it also stores couples seen in validation or testing
def _compute_training_couples(dataset:Dataset):
    head_rel_training_couples = set()
    rel_tail_training_couples = set()

    for current_sample in dataset.train_samples:
        head_rel_training_couples.add((current_sample[0], current_sample[1]))
        rel_tail_training_couples.add((current_sample[1], current_sample[2]))

    return head_rel_training_couples, rel_tail_training_couples


def cossim(a, b):
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()

    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))


# TODO: VERY TEMPORARY
def read_entity_types(dataset: Dataset):
    entity_2_types = defaultdict(lambda: set())
    type_2_entities = defaultdict(lambda: set())

    filepath = os.path.join(dataset.home, "entity_types.csv")
    with open(filepath, "r") as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            entity_name, type_name = line.strip().split("\t")
            entity_id = dataset.entity_name_2_id[entity_name]
            entity_2_types[entity_id].add(type_name)
            type_2_entities[type_name].add(entity_id)

    return entity_2_types, type_2_entities
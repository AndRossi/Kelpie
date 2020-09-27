import numpy

from dataset import Dataset
from model import Model


def extract_comparable_entities(model: Model,
                                dataset: Dataset,
                                sample: numpy.array,
                                perspective: str,
                                convertibility_param: float):

    head, relation, tail = sample
    entity_to_explain, target_entity = (head, tail) if perspective == "head" else (tail, head)

    # extract the ids of all entities for which
    #   <entity, relation, tail> is not present in the training set if perspective is head
    # or
    #   <head, relation, entity> is not present in the training set if perspective is tail
    other_entities = []
    samples = []
    if perspective == "head":
        for cur_entity in range(0, dataset.num_entities):
            if cur_entity == entity_to_explain or dataset.to_filter(cur_entity, relation, tail):
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

    #entity_to_explain_score = model.score(samples=sample)[0]
    samples_all_scores = model.all_scores(samples=numpy.array(samples)).detach().cpu().numpy()

    other_entity_2_resistance = {}
    for i in range(len(other_entities)):
        current_other_entity = other_entities[i]

        current_sample_all_scores = samples_all_scores[i]
        current_sample_best_score = numpy.max(current_sample_all_scores)
        current_sample_target_score = current_sample_all_scores[target_entity]

        current_other_entity_resistance_to_conversion = current_sample_best_score-current_sample_target_score

        # if resistance to conversion is 0, current_sample ranked in 1st position the target entity,
        # so the current_other_entity is useless for our purpose
        # (we want current_other_entities that, without additions, do not predict the target entity with rank 1)
        # (e.g. if we are explaining <Barack, nationality, USA>, Xin is an acceptable "convertible entity"
        # only if <Xin, nationality, ?> does not already rank USA in 1st position!)
        if current_other_entity_resistance_to_conversion <= 0:
            continue
        else:
            other_entity_2_resistance[current_other_entity] = current_other_entity_resistance_to_conversion

    # sort by resistance in ascending order
    other_entity_resistance_couples = sorted(other_entity_2_resistance.items(), key=lambda x: x[1], reverse=False)

    # return the first X% couples, where X is the convertibility_param
    output_size = int(len(other_entity_resistance_couples)*convertibility_param)
    return other_entity_resistance_couples[:output_size]


def extract_entities_by_proximity(model: Model,
                                  dataset: Dataset,
                                  entity_id: int,
                                  proximity_param: float):

    all_entity_embeddings = model.entity_embeddings.detach().cpu().numpy()
    entity_embedding = all_entity_embeddings[entity_id, :]

    entity_distances = []
    for other_entity in dataset.entities:
        other_entity_embedding = all_entity_embeddings[other_entity, :]
        entity_distances.append((other_entity, numpy.linalg.norm(entity_embedding-other_entity_embedding, ord=2)))

    entity_distances = sorted(entity_distances, key=lambda x: x[1], reverse=False)

    # return the first X% couples, where X is the convertibility_param
    output_size = int(len(entity_distances)*proximity_param)
    return entity_distances[:output_size]

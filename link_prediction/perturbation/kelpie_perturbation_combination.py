import math
import random

import numpy

from heuristics import relevance_estimate


def remove_combination(dataset,
                       sample_to_explain,       # the sample to explain
                       samples: numpy.array,    # training samples to perturbate (each one does contain the kelpie entity),
                       kelpie_entity_id,        # the kelpie entity
                       n=2,                     # number of samples to affect in each perturbation (couples, triples, etc)
                       budget=100):             # maximum number of perturbations to generate and return
    output = []
    skipped = []

    relation_2_head_rel_tfidf, relation_2_tail_rel_tfidf = relevance_estimate.read(dataset)

    head_id, relation_id, tail_id = sample_to_explain
    relation = dataset.relation_id_2_name[relation_id]

    sample_estimates = []

    if head_id == kelpie_entity_id:
        for (cur_head_id, cur_relation_id, cur_tail_id) in samples:
            cur_relation = dataset.relation_id_2_name[cur_relation_id]
            head_relation = cur_relation if cur_head_id == kelpie_entity_id else "INV_" + cur_relation
            sample_estimates.append(((cur_head_id, cur_relation_id, cur_tail_id), relation_2_head_rel_tfidf[relation][head_relation]))
    else:
        for (cur_head_id, cur_relation_id, cur_tail_id) in samples:
            cur_relation = dataset.relation_id_2_name[cur_relation_id]
            tail_relation = cur_relation if cur_tail_id == kelpie_entity_id else "INV_" + cur_relation
            sample_estimates.append(((cur_head_id, cur_relation_id, cur_tail_id), relation_2_tail_rel_tfidf[relation][tail_relation]))

    sample_estimates = sorted(sample_estimates, key= lambda x: x[1], reverse=True)

    # compute the number k of the top facts to include, based on n and on the budget
    if n == 2:
        k = math.floor(math.sqrt(budget))
    elif n == 3:
        k = math.floor(numpy.cbrt(budget))
    else:
        raise NotImplementedError

    # get the top facts to include
    samples_to_combine = set([sample_estimate[0] for sample_estimate in sample_estimates[0:k]])

    # get the indices of these facts in the original numpy array of samples
    sample_to_combine_2_index = {}
    for i in range(len(samples)):
        cur_sample = tuple(samples[i])
        if cur_sample in samples_to_combine:
            sample_to_combine_2_index[cur_sample] = i

    if n == 2:
        for s1 in samples_to_combine:
            for s2 in samples_to_combine:
                if s2 == s1:
                    continue
                output.append(numpy.delete(samples,
                                           [sample_to_combine_2_index[s1], sample_to_combine_2_index[s2]],
                                           axis=0))
                skipped.append([s1, s2])
    elif n == 3:
        for s1 in samples_to_combine:
             for s2 in samples_to_combine:
                if s2 == s1:
                    continue

                for s3 in samples_to_combine:
                    if s3 == s1 or s3 == s2:
                        continue
                    output.append(numpy.delete(samples,
                                               [sample_to_combine_2_index[s1],
                                                sample_to_combine_2_index[s2],
                                                sample_to_combine_2_index[s3]],
                                               axis=0))
                    skipped.append([s1, s2, s3])
    else:
        raise NotImplementedError

    return output, skipped


def remove_combination_random(dataset,
                              sample_to_explain,       # the sample to explain
                              samples: numpy.array,    # training samples to perturbate (each one does contain the kelpie entity),
                              kelpie_entity_id,           # the kelpie entity
                              n=2,                     # number of samples to affect in each perturbation (couples, triples, etc)
                              budget=100):    # all the samples containing the kelpie entity

    output = []
    skipped = []

    i = 0
    if n == 2:
        while i < budget:
            i1 = random.randint(0, len(samples)-1)
            i2 = random.randint(0, len(samples)-1)
            while i2 == i1:
                i2 = random.randint(0, len(samples) - 1)

            (i1, i2) = sorted((i1, i2))
            if (samples[i1], samples[i2]) in skipped:
                continue

            output.append(numpy.delete(samples, [i1, i2], axis=0))
            skipped.append([samples[i1], samples[i2]])
            i+= 1

    elif n == 3:
        while i < budget:
            i1 = random.randint(0, len(samples) - 1)
            i2 = random.randint(0, len(samples) - 1)
            while i2 == i1:
                i2 = random.randint(0, len(samples) - 1)
            i3 = random.randint(0, len(samples) - 1)
            while i3 == i1 or i3 == i2:
                i3 = random.randint(0, len(samples) - 1)

            (i1, i2, i3) = sorted((i1, i2, i3))
            if (samples[i1], samples[i2], samples[i3]) in skipped:
                continue

            output.append(numpy.delete(samples, [i1, i2, i3], axis=0))
            skipped.append([samples[i1], samples[i2], samples[i3]])
            i += 1

    else:
        raise NotImplementedError

    return output, skipped
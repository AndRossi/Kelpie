import os
from collections import defaultdict
from multiprocessing.pool import Pool

import numpy

from config import MAX_PROCESSES
from dataset import Dataset, FB15K

def compute(dataset: Dataset):
    """
        for each relation, compute the TF-IDF vector of all relation paths co-occurring with those relations
        :param dataset:
        :return:
    """

    entity_2_samples = {entity: [] for entity in dataset.entities}
    relation_2_samples = {relation: [] for relation in dataset.relations}

    all_triples = numpy.vstack((dataset.train_triples, dataset.valid_triples, dataset.test_triples))
    for (h, r, t) in all_triples:
        entity_2_samples[h].append((h, r, t))
        entity_2_samples[t].append((h, r, t))
        relation_2_samples[r].append((h, r, t))

    relation_2_relpath_counts = defaultdict(lambda: defaultdict(lambda:0))
    relation_2_relpathset = defaultdict(lambda: set())    # this is just a gimmick to improve efficiency in STEP 2
    all_relpaths = set()

    # STEP 1: count the co-occurrences of all relation paths with each relation
    print("Extracting relational paths for all relations...")

    relations_list = list(dataset.relations)

    relation_2_count = {}
    worker_processes_inputs = []

    for i, relation in enumerate(relations_list):
        relation_2_count[relation] = len(relation_2_samples[relation])
        worker_processes_inputs.append((i, relation, relation_2_samples[relation], entity_2_samples))

    with Pool(processes=MAX_PROCESSES) as pool:
        result = pool.map(multiprocess_extract_relpaths, worker_processes_inputs)
        for i, relation in enumerate(relations_list):
            relpaths_under_relation_2_count = result[i]
            relation_2_relpath_counts[relation] = relpaths_under_relation_2_count
            relation_2_relpathset[relation] = set(relpaths_under_relation_2_count.keys())
            for x in relation_2_relpathset[relation]:
                all_relpaths.add(x)

    # STEP 2: COMPUTE DOCUMENT FREQUENCIES
    # that is, for each "word" (relpath), the number of "documents" (relations) it appears in
    # we rely on relation_2_relpathset, which uses sets and can be accessed with O(1) computational complexity
    print("Computing DFs...")
    relpath_2_df = {}
    for relpath in all_relpaths:
        df = 0
        for r in dataset.relations:
            if relpath in relation_2_relpathset[r]:
                df += 1
        relpath_2_df[relpath] = df

    # STEP 3: COMPUTE INVERSE DOCUMENT FREQUENCIES
    # that is, for each "word", compute log( |all documents| / DF[word] )
    # IDF is 0 if a "word" does not appear in any "document"
    print("Computing IDFs...")
    relpath_2_idf = {}
    for relpath in all_relpaths:
        relpath_2_idf[relpath] = numpy.log(float(len(dataset.relations) / relpath_2_df[relpath]))

    # STEP 4: COMPUTE TERM FREQUENCIES
    # that is, for each couple of "word" and "document", the ratio between
    #       the number of times that the "word" appears in the "document"
    #       the number of times any word appears in the "document"
    print("Computing TFs...")
    relation_2_relpath_tf = defaultdict(lambda: {})
    for relation in dataset.relations:
        relpath_counts = relation_2_relpath_counts[relation]

        denominator = 0.0
        for x in relpath_counts:
            denominator += relpath_counts[x]

        for relpath in all_relpaths:
            numerator = float(relpath_counts[relpath] if relpath in relpath_counts else 0.0)

            if denominator != 0:
                tf = numerator / denominator
            else:
                tf = 0

            relation_2_relpath_tf[relation][relpath] = tf

    # STEP 5: COMPUTE TF-IDF WEIGHTS
    # that is, for each couple of "word" and "document", the product TF[word, document] * IDF[word]
    relation_2_relpath_tfidf = defaultdict(lambda: {})
    for relation in dataset.relations:
        for relpath in all_relpaths:
            tfidf = relation_2_relpath_tf[relation][relpath] * relpath_2_idf[relpath]
            relation_2_relpath_tfidf[relation][relpath] = tfidf

    return all_relpaths, relation_2_relpath_tfidf

def read(dataset: Dataset):
    print("Reading %s tfidf weights for relevance estimate..." % dataset.name)
    headrel_file = os.path.join(dataset.home, dataset.name + "_head_tfidf.csv")
    tailrel_file = os.path.join(dataset.home, dataset.name + "_tail_tfidf.csv")

    relation_2_head_rel_tfidf = defaultdict(lambda: defaultdict(lambda:0.0))
    relation_2_tail_rel_tfidf = defaultdict(lambda: defaultdict(lambda:0.0))

    with open(headrel_file) as inputfile:
        for line in inputfile.readlines():
            relation, head_tfidf = line.strip().split("\t")
            head_tfidf_strings = head_tfidf[1:-1].split(", ")

            for x in head_tfidf_strings:
                head_rel, tfidf_value = x.split(":")
                relation_2_head_rel_tfidf[relation][head_rel] = float(tfidf_value)

    with open(tailrel_file) as inputfile:
        for line in inputfile.readlines():
            relation, tail_tfidf = line.strip().split("\t")
            tail_tfidf_strings = tail_tfidf[1:-1].split(", ")

            for x in tail_tfidf_strings:
                tail_rel, tfidf_value = x.split(":")
                relation_2_tail_rel_tfidf[relation][tail_rel] = float(tfidf_value)

    return relation_2_head_rel_tfidf, relation_2_tail_rel_tfidf

def save(dataset):
    all_relpaths, relation_2_relpath_tfidf = compute(dataset)

    output_file = os.path.join(dataset.home, dataset.name + "_tfidf.csv")
    lines = []
    with open(output_file, "w") as outputfile:
        for relation in dataset.relations:
            line = relation

            relpath_2_tfidf = relation_2_relpath_tfidf[relation]
            relpath_tfidf_couples = list(relpath_2_tfidf.items())

            # sort by descending tfidf value, and only take into account the top 500 relpaths
            relpath_tfidf_couples.sort(key=lambda x: x[1], reverse=True)
            relpath_tfidf_couples = relpath_tfidf_couples[:500]

            tfidf_values = []
            for relpath_tfidf_couple in relpath_tfidf_couples:
                relpath, value = relpath_tfidf_couple
                if value > 0:
                    tfidf_values.append(relpath + ":" + str(value))

            line += "\t[" + ", ".join(tfidf_values) + "]\n"
            lines.append(line)
        outputfile.writelines(lines)


def multiprocess_extract_relpaths(input_data):
    i, relation, samples_featuring_relation, entity_2_samples = input_data

    print("\t%i: %s" % (i, relation))
    relpath_2_count = dict()
    for sample in samples_featuring_relation:
        
        one_step_relpaths = extract_one_step_relpaths(sample, entity_2_samples)
        for one_step_relpath in one_step_relpaths:
            if one_step_relpath in relpath_2_count:
                relpath_2_count[one_step_relpath] += 1
            else:
                relpath_2_count[one_step_relpath] = 1

        two_step_relpaths = extract_two_step_relpaths(sample, entity_2_samples)
        for two_step_relpath in two_step_relpaths:
            if two_step_relpath in relpath_2_count:
                relpath_2_count[two_step_relpath] += 1
            else:
                relpath_2_count[two_step_relpath] = 1

        three_step_relpaths = extract_three_step_relpaths(sample, entity_2_samples)
        for three_step_relpath in three_step_relpaths:
            if three_step_relpath in relpath_2_count:
                relpath_2_count[three_step_relpath] += 1
            else:
                relpath_2_count[three_step_relpath] = 1

    return relpath_2_count

def extract_one_step_graph_paths(sample, entity_2_samples):
    head, relation, tail = sample

    one_step_graph_paths = []

    # extract all samples containing the head
    head_samples = entity_2_samples[head]
    for head_sample in head_samples:
        cur_head, cur_relation, cur_tail = head_sample

        # ignore the original fact itself
        if cur_head == head and cur_relation == relation and cur_tail == tail:
            continue

        # note: the input fact can be a self-loop (<h, r, h>)
        # the 1-step path facts, in this case, will obviously be self-loops too

        # extract one-step paths
        if cur_head == head and cur_tail == tail:
            one_step_graph_paths.append((cur_head, cur_relation, cur_tail))
        elif cur_head == tail and cur_tail == head:
            one_step_graph_paths.append((cur_tail, "INV_" + cur_relation, cur_head))

    # TODO: this is just a control. It can be commented after we are sure the method works.
    for one_step_graph_path in one_step_graph_paths:
        (e1, r1, e2) = one_step_graph_path
        assert e1 == head and e2 == tail

    return one_step_graph_paths

def extract_two_step_graph_paths(sample, entity_2_samples):
    head, relation, tail = sample
    two_step_graph_paths = set()
    head_samples = entity_2_samples[head]

    # iterate over all samples <cur_head, cur_relation, cur_tail> containing the head:
    # these are potential first steps for the two-step paths we are searching
    for head_sample in head_samples:
        cur_head, cur_relation, cur_tail = head_sample

        # ignore self-loops at all
        if cur_head == cur_tail:
            continue

        # ignore one-step paths
        elif cur_head == head and cur_tail == tail or cur_tail == head and cur_head == tail:
            continue

        # if head is cur_head, we need to find all one-step paths connecting cur_tail and tail
        elif cur_head == head:
            one_step_graph_paths = extract_one_step_graph_paths((cur_tail, None, tail), entity_2_samples)
            for one_step_graph_path in one_step_graph_paths:
                two_step_graph_paths.add(((cur_head, cur_relation, cur_tail), one_step_graph_path))

        # elseif head is cur_tail, we need to find all one-step paths connecting cur_head and tail
        elif cur_tail == head:
            one_step_graph_paths = extract_one_step_graph_paths((cur_head, None, tail), entity_2_samples)
            for one_step_graph_path in one_step_graph_paths:
                two_step_graph_paths.add(((cur_tail, "INV_" + cur_relation, cur_head), one_step_graph_path))

    # TODO: this is just a control. It can be commented after we are sure the method works.
    for two_step_graph_path in two_step_graph_paths:
        (e1, _, e2a), (e2b, _, e3) = two_step_graph_path
        assert e2a == e2b and e1 == head and e3 == tail
    return two_step_graph_paths


def extract_three_step_graph_paths(sample, entity_2_samples):
    head, relation, tail = sample

    three_step_graph_paths = set()
    head_samples = entity_2_samples[head]

    # iterate over all samples <cur_head, cur_relation, cur_tail> containing the head:
    # these are potential first steps for the two-step relational paths we are searching

    for head_sample in head_samples:
        cur_head, cur_relation, cur_tail = head_sample

        # ignore the case in which the first step is a self-loop
        if cur_head == cur_tail:
            continue

        # ignore the case in which the first step leads to the tail directly
        elif cur_head == head and cur_tail == tail or cur_tail == head and cur_head == tail:
            continue

        # if head is cur_head, we need to find all two-step paths connecting cur_tail and tail
        elif cur_head == head:
            two_step_graph_paths = extract_two_step_graph_paths((cur_tail, None, tail), entity_2_samples)
            for two_step_graph_path in two_step_graph_paths:
                three_step_graph_paths.add(((cur_head, cur_relation, cur_tail), two_step_graph_path[0], two_step_graph_path[1]))

        # else if head is cur_tail, we need to find all two-step paths connecting cur_head and tail
        elif cur_tail == head:
            two_step_graph_paths = extract_two_step_graph_paths((cur_head, None, tail), entity_2_samples)
            for two_step_graph_path in two_step_graph_paths:
                three_step_graph_paths.add(((cur_tail, "INV_" + cur_relation, cur_head), two_step_graph_path[0], two_step_graph_path[1]))


    # In three step paths, ideally we want a structure such as e1 ---> e2 ---> e3 ---> e4
    # possible dysfunctions:
    #   e1 ---> e1 ---> e3 ---> e4      this is avoided because we explicitly ignore self-loops among head samples
    #   e1 ---> e2 ---> e3 ---> e1      this is not a dysfunction (the original sample can be a self-loop)
    #   e1 ---> e2 ---> e1 ---> e3      THIS IS POSSIBLE AND WE ARE NOT PROTECTED AGAINST THIS.
    # we explicitly search for e1 ---> e2 ---> e1 ---> e3 paths and remove them:

    filtered_three_step_graph_paths = set()

    for three_step_graph_path in three_step_graph_paths:
        (e1, _, e2a), (e2b, _, e3a), (e3b, _, e4) = three_step_graph_path
        assert e2a == e2b and e3a == e3b and e1 == head and e4 == tail

        if e1 == e3a:
            continue

        filtered_three_step_graph_paths.add(three_step_graph_path)

    return filtered_three_step_graph_paths



def extract_one_step_relpaths(sample, entity_2_samples):
    one_step_graph_paths = extract_one_step_graph_paths(sample, entity_2_samples)
    one_step_relpaths = set()
    for one_step_graph_path in one_step_graph_paths:
        path_head, path_relation, path_tail = one_step_graph_path
        one_step_relpaths.add(path_relation)
    return one_step_relpaths

def extract_two_step_relpaths(sample, entity_2_samples):
    two_step_graph_paths = extract_two_step_graph_paths(sample, entity_2_samples)
    two_step_relpaths = set()
    for two_step_graph_path in two_step_graph_paths:
        (e1, r1, e2a), (e2b, r2, e3) = two_step_graph_path
        two_step_relpaths.add(";".join([r1, r2]))
    return two_step_relpaths

def extract_three_step_relpaths(sample, entity_2_samples):
    three_step_graph_paths = extract_three_step_graph_paths(sample, entity_2_samples)
    three_step_relpaths = set()
    for three_step_graph_path in three_step_graph_paths:
        (e1, r1, e2a), (e2b, r2, e3a), (e3b, r3, e4) = three_step_graph_path
        three_step_relpaths.add(";".join([r1, r2, r3]))
    return three_step_relpaths


save(Dataset(FB15K))
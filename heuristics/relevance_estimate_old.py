import os
from collections import defaultdict

import numpy

from kelpie.dataset import Dataset, FB15K, WN18, WN18RR, FB15K_237, YAGO3_10


def compute(dataset: Dataset):
    entity_2_samples = defaultdict(lambda: [])
    relation_2_samples = defaultdict(lambda: [])

    all_triples = numpy.vstack((dataset.train_triples, dataset.valid_triples, dataset.test_triples))
    for (h, r, t) in all_triples:
        entity_2_samples[h].append((h, r, t))
        entity_2_samples[t].append((h, r, t))
        relation_2_samples[r].append((h, r, t))

    relation_2_occurrences = {}
    relation_2_from_head_rels = defaultdict(lambda: defaultdict(lambda: 0))
    relation_2_to_tail_rels = defaultdict(lambda: defaultdict(lambda: 0))

    # for each relation, compute the TF-IDF vector of all other relations
    i=0
    for relation in dataset.relations:
        i+=1
        print("Extracting head_rels and tail_rels for relation %i: %s" % (i, relation))
        samples = relation_2_samples[relation]     # all training samples containing r
        relation_2_occurrences[relation] = len(samples)

        for (head, _, tail) in samples:

            samples_containing_head = entity_2_samples[head]
            samples_containing_tail = entity_2_samples[tail]

            from_head_rels = set()
            to_tail_rels = set()

            for sample_containing_head in samples_containing_head:
                if sample_containing_head == (head, relation, tail):
                    continue
                (x, y, z) = sample_containing_head
                from_head_rel = y if x == head else "INV_" + y
                from_head_rels.add(from_head_rel)

            for sample_containing_tail in samples_containing_tail:
                if sample_containing_tail == (head, relation, tail):
                    continue
                (x, y, z) = sample_containing_tail
                to_tail_rel = y if z == tail else "INV_" + y
                to_tail_rels.add(to_tail_rel)

            for from_head_rel in from_head_rels:
                relation_2_from_head_rels[relation][from_head_rel] += 1
            for to_tail_rel in to_tail_rels:
                relation_2_to_tail_rels[relation][to_tail_rel] += 1


    # STEP 2: COMPUTE DOCUMENT FREQUENCIES
    # that is, for each "word" (from_head_rel or to_tail_rel), the number of "documents" (relations) it appears in
    head_rel_2_df = defaultdict(lambda: 0)
    for document in relation_2_from_head_rels:
        for word in relation_2_from_head_rels[document]:
            head_rel_2_df[word] +=1

    tail_rel_2_df = defaultdict(lambda: 0)
    for document in relation_2_to_tail_rels:
        for word in relation_2_to_tail_rels[document]:
            tail_rel_2_df[word] +=1

    # STEP 3: COMPUTE INVERSE DOCUMENT FREQUENCIES
    # that is, for each "word", compute log( |all documents| / DF[word] )
    # IDF is 0 if a "word" does not appear in any "document"
    head_rel_2_idf = defaultdict(lambda: 0)
    for head_rel in head_rel_2_df:
        head_rel_2_idf[head_rel] = numpy.log(float(dataset.num_relations)/float(head_rel_2_df[head_rel]))

    tail_rel_2_idf = defaultdict(lambda: 0)
    for tail_rel in tail_rel_2_df:
        tail_rel_2_idf[tail_rel] = numpy.log(float(dataset.num_relations)/float(tail_rel_2_df[tail_rel]))


    # STEP 4: COMPUTE TERM FREQUENCIES
    # that is, for each couple of "word" and "document", the ratio between
    #       the number of times that the "word" appears in the "document"
    #       the number of times any word appears in the "document"
    relation_2_head_rel_tf = defaultdict(lambda: defaultdict(lambda:0.0))
    for relation in dataset.relations:
        head_rel_2_occurrences = relation_2_from_head_rels[relation]
        denominator = numpy.sum(list(head_rel_2_occurrences.values()))
        for head_rel in head_rel_2_occurrences:
            numerator = head_rel_2_occurrences[head_rel]
            relation_2_head_rel_tf[relation][head_rel] = float(numerator)/float(denominator)

    relation_2_tail_rel_tf = defaultdict(lambda: defaultdict(lambda:0.0))
    for relation in dataset.relations:
        tail_rel_2_occurrences = relation_2_to_tail_rels[relation]
        denominator = numpy.sum(list(tail_rel_2_occurrences.values()))
        for tail_rel in tail_rel_2_occurrences:
            numerator = tail_rel_2_occurrences[tail_rel]
            relation_2_tail_rel_tf[relation][tail_rel] = float(numerator)/float(denominator)

    # STEP 5: COMPUTE TF-IDF WEIGHTS
    # that is, for each couple of "word" and "document", the product TF[word, document] * IDF[word]
    relation_2_head_rel_tfidf = defaultdict(lambda: defaultdict(lambda:0.0))
    relation_2_tail_rel_tfidf = defaultdict(lambda: defaultdict(lambda:0.0))
    for relation in dataset.relations:
        for other_rel in [x for x in dataset.relations] + ["INV_" + x for x in dataset.relations]:
            relation_2_head_rel_tfidf[relation][other_rel] = relation_2_head_rel_tf[relation][other_rel] * head_rel_2_idf[other_rel]
            relation_2_tail_rel_tfidf[relation][other_rel] = relation_2_tail_rel_tf[relation][other_rel] * tail_rel_2_idf[other_rel]

    return relation_2_head_rel_tfidf, relation_2_tail_rel_tfidf


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
    relation_2_head_tfidf, relation_2_tail_tfidf = compute(dataset)

    headrel_file = os.path.join(dataset.home, dataset.name + "_head_tfidf.csv")
    tailrel_file = os.path.join(dataset.home, dataset.name + "_tail_tfidf.csv")

    vocabulary = [x for x in dataset.relations] + ["INV_" + x for x in dataset.relations]


    with open(headrel_file, "w") as outputfile:
        lines = []
        for relation in dataset.relations:
            line = relation
            print(relation)
            headrel_2_tfidf = relation_2_head_tfidf[relation]

            tfidf_values = []
            for head_rel in vocabulary:
                tfidf_values.append(head_rel + ":" + str(headrel_2_tfidf[head_rel]))

            line += "\t[" + ", ".join(tfidf_values) + "]\n"
            lines.append(line)
        outputfile.writelines(lines)

    with open(tailrel_file, "w") as outputfile:
        lines = []
        for relation in dataset.relations:
            line = relation
            print(relation)
            tailrel_2_tfidf = relation_2_tail_tfidf[relation]

            tfidf_values = []
            for tail_rel in vocabulary:
                tfidf_values.append(tail_rel + ":" + str(tailrel_2_tfidf[tail_rel]))

            line += "\t[" + ", ".join(tfidf_values) + "]\n"
            lines.append(line)
        outputfile.writelines(lines)
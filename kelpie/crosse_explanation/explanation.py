import html
import os
from collections import defaultdict

import numpy as np


TOP_RELATIONS_K = 10
TOP_ENTITIES_K = 3

class CrossEExplanation:

    def __init__(self, dataset, model, head, relation, tail):
        self.dataset = dataset
        self.model = model

        self.paths_folder = os.path.join(self.dataset.home, "paths")

        print("Reading one step paths co-occurring with test facts...")
        self.test_fact_2_one_step_paths = self._read_paths(os.path.join(self.paths_folder, "test_facts_with_one_step_graph_paths.csv"))
        print("Reading two step paths co-occurring with test facts...")
        self.test_fact_2_two_step_paths = self._read_paths(os.path.join(self.paths_folder, "test_facts_with_two_step_graph_paths.csv"))
        print("Reading one step paths co-occurring with train facts...")
        self.train_fact_2_one_step_paths = self._read_paths(os.path.join(self.paths_folder, "train_facts_with_one_step_graph_paths.csv"))
        print("Reading two step paths co-occurring with train facts...")
        self.train_fact_2_two_step_paths = self._read_paths(os.path.join(self.paths_folder, "train_facts_with_two_step_graph_paths.csv"))

        self.model = model
        self.head, self.relation, self.tail = head, relation, tail
        self.head_id, self.relation_id, self.tail_id = self.dataset.entity_name_2_id[head], \
                                                       self.dataset.relation_name_2_id[relation], \
                                                       self.dataset.entity_name_2_id[tail]

        self.fact = (self.head, self.relation, self.tail)
        self.tuple = (self.head_id, self.relation_id, self.tail_id)


    def run(self):

        # get paths for the fact to explain
        one_step_paths = self.test_fact_2_one_step_paths[(self.head, self.relation, self.tail)]
        two_step_paths = self.test_fact_2_two_step_paths[(self.head, self.relation, self.tail)]

        # get top relations similar to r: similar_relations = {rs_1, rs_2 ... rs_k}
        all_relation_embeddings = self.model.relation_embeddings.cpu().numpy()
        top_similar_relation_ids = self.top_similar_embeddings(self.relation_id, all_relation_embeddings, TOP_RELATIONS_K)
        top_similar_relation_names = [self.dataset.relation_id_2_name[x] for x in top_similar_relation_ids]

        # get all paths that connect the head and the tail
        # and in which either the first relation in top_similar_relation_ids
        # or the inverse of the first relation is in top_similar_relation_ids
        one_step_filtered_paths = []
        two_step_filtered_paths = []
        for path in one_step_paths:
            (h, r, t) = path
            r_inv = self._invert_relation_name(r)
            if r in top_similar_relation_names or r_inv in top_similar_relation_names:
                one_step_filtered_paths.append(path)
        for path in two_step_paths:
            (h, r, _, _, _, _) = path
            r_inv = self._invert_relation_name(r)
            if r in top_similar_relation_names or r_inv in top_similar_relation_names:
                one_step_filtered_paths.append(path)

        # get top entities similar to head: similar_entities = {es_1, es_2 ... es_k}
        all_entity_embeddings = self.model.entity_embeddings.cpu().numpy()
        top_similar_entity_ids = self.top_similar_embeddings(self.head_id, all_entity_embeddings, TOP_ENTITIES_K)
        top_similar_entity_names = [self.dataset.entity_id_2_name[x] for x in top_similar_entity_ids]

        similar_fact_2_one_step_paths = defaultdict(lambda: [])
        similar_fact_2_two_step_paths = defaultdict(lambda: [])
        for cur_head in top_similar_entity_names:
            for cur_tail in self.dataset.entities:
                cur_fact = (cur_head, self.relation, cur_tail)
                if cur_fact in self.train_fact_2_one_step_paths:
                    similar_fact_2_one_step_paths[cur_fact] = self.train_fact_2_one_step_paths[cur_fact]
                if cur_fact in self.train_fact_2_two_step_paths:
                    similar_fact_2_two_step_paths[cur_fact] = self.train_fact_2_two_step_paths[cur_fact]


        # map each path in one_step and two_step path to its "support", that is, the number of times
        # that, in the training set, similar paths co-occur with the same relation
        one_step_path_2_support = {x: 0.0 for x in one_step_filtered_paths}
        two_step_path_2_support = {x: 0.0 for x in two_step_filtered_paths}

        # Per ogni relation_path che collega h a t segnati in quante altre hs lo stesso relation path collega a ts.
        # Quello sarà il supporto di quel relation_path
        for (head, cur_relation, tail) in one_step_filtered_paths:
            for similar_fact in similar_fact_2_one_step_paths:
                for (_, other_relation, _) in similar_fact_2_one_step_paths[similar_fact]:
                    if cur_relation == other_relation:
                        one_step_path_2_support[(head, cur_relation, tail)] += 1

        for (head, cur_relation_1, e1, e1, cur_relation_2, e2) in two_step_filtered_paths:
            for similar_fact in similar_fact_2_two_step_paths:
                for (_, other_relation_1, _, _, other_relation_2, _) in similar_fact_2_one_step_paths[similar_fact]:
                    if cur_relation_1 == other_relation_1 and cur_relation_2 == other_relation_2:
                        two_step_path_2_support[(head, cur_relation_1, e1, e1, cur_relation_2, e2)] += 1

        one_step_path_supports = sorted(one_step_path_2_support.items(), key= lambda x:x[1])
        two_step_path_supports = sorted(one_step_path_2_support.items(), key= lambda x:x[1])

        print(one_step_path_supports[:10])
        print(two_step_path_supports[:10])

    def _read_paths(self, filepath):

        fact_2_paths = dict()
        with open(filepath, "r") as input_file:

            for line in input_file:

                line = html.unescape(line.strip())

                head_name, relation_name, tail_name, paths_str = line.split(";", 3)
                paths_str = paths_str[1:-1]
                paths = list()
                if len(paths_str) > 0:
                    for path_str in paths_str.split("|"):
                        paths.append(path_str.split(";"))

                fact_2_paths[(head_name, relation_name, tail_name)] = paths

        return fact_2_paths


    def _invert_relation_name(self, relation_name:str):
        if relation_name.startswith("INVERSE_"):
            return relation_name[8:]
        else:
            return "INVERSE_" + relation_name

    def top_similar_embeddings(self,
                                embedding_id: int,
                                embedding_list:np.array,
                                k: int):

        id_2_distance = {}
        for i in range(embedding_list.shape[0]):
            if i == embedding_id:
                continue

            distance = np.linalg.norm(embedding_list[embedding_id] - embedding_list[i], 2)
            id_2_distance[i] = distance

        return sorted(id_2_distance.items(), key=lambda x:x[1])[:k]
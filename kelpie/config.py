import functools
import os

import torch

ROOT = os.path.abspath("/Users/andrea/kelpie")
MODEL_PATH = os.path.join(ROOT, "stored_models")
MAX_PROCESSES = 4
#ROOT = os.path.abspath("/home/nvidia/workspace/dbgroup/andrea/kelpie")
#ROOT = os.path.abspath("/home/nvidia/workspace/dbgroup/agiw/gruppo1/Kelpie")


from scipy import stats
list1 = ["a", "b", "c"]
list2 = ["d", "e", "f"]


item_2_rank = {"a":1, "b":2, "c": 3, "d": 4, "e": 5, "f": 6}
def spearman(a, b):

    ranked_a = [item_2_rank[x] for x in a]
    ranked_b = [item_2_rank[x] for x in b]

    return stats.kendalltau(ranked_a, ranked_b)

print(spearman(list1, list2))
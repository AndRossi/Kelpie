"""
This module does explanation for ComplEx model
"""
import argparse
import os

import torch
import numpy

from kelpie import kelpie_perturbation, ranking_similarity, config
from kelpie.data_poisoning import data_poisoning_relevance
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import KelpieEvaluator
from kelpie.kelpie_dataset import KelpieDataset
from kelpie.models.complex.model import ComplEx, KelpieComplEx
from kelpie.models.complex.optimizer import KelpieComplExOptimizer

def read(filepath):
    fact_2_most_relevant_facts = {}
    with open(filepath, "r") as inputfile1:
        inputlines1 = inputfile1.readlines()
        for line in inputlines1:
            fact_to_explain, most_relevant_facts = line.strip().split("[")
            most_relevant_facts = most_relevant_facts[:-1]
            most_relevant_facts = most_relevant_facts.split(",")
            fact_2_most_relevant_facts[fact_to_explain] = most_relevant_facts
    return fact_2_most_relevant_facts

kelpie_input_filepath = os.path.join(config.ROOT, "output_1.txt")
dt_input_filepath = os.path.join(config.ROOT, "output_2.txt")

kelpie__fact_2_most_relevant_facts = read(kelpie_input_filepath)
dt__fact_2_most_relevant_facts = read(dt_input_filepath)


for fact in kelpie__fact_2_most_relevant_facts:
    print(fact + ": " + str(ranking_similarity.rank_biased_overlap(kelpie__fact_2_most_relevant_facts[fact],
                                                                   dt__fact_2_most_relevant_facts[fact]
                                                                   )))
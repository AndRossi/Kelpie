import sys
import os
import argparse
import random
import time

import numpy
import torch

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

from dataset import Dataset, ALL_DATASET_NAMES
from kelpie import Kelpie as Kelpie
from data_poisoning import DataPoisoning
from criage import Criage
from link_prediction.models.transe import TransE
from link_prediction.models.model import BATCH_SIZE, LEARNING_RATE, EPOCHS, DIMENSION, MARGIN, NEGATIVE_SAMPLES_RATIO, \
    REGULARIZER_WEIGHT
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER

parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
                    type=str,
                    choices=ALL_DATASET_NAMES,
                    help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")

parser.add_argument("--max_epochs",
                    type=int,
                    default=1000,
                    help="Number of epochs.")

parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Batch size.")

parser.add_argument("--learning_rate",
                    type=float,
                    default=0.0005,
                    help="Learning rate.")

parser.add_argument("--dimension",
                    type=int,
                    default=200,
                    help="Embedding dimensionality.")

parser.add_argument("--margin",
                    type=int,
                    default=5,
                    help="Margin for pairwise ranking loss.")

parser.add_argument("--negative_samples_ratio",
                    type=int,
                    default=3,
                    help="Number of negative samples for each positive sample.")

parser.add_argument("--regularizer_weight",
                    type=float,
                    default=0.0,
                    help="Weight for L2 regularization.")

parser.add_argument("--model_path",
                    type=str,
                    help="Path where to find the model to explain")

parser.add_argument("--facts_to_explain_path",
                    type=str,
                    help="Path where to find the facts to explain")

parser.add_argument("--coverage",
                    type=int,
                    default=10,
                    help="Number of random entities to extract and convert")

parser.add_argument("--baseline",
                    type=str,
                    default=None,
                    choices=[None, "k1", "data_poisoning", "criage"],
                    help="attribute to use when we want to use a baseline rather than the Kelpie engine")

parser.add_argument("--entities_to_convert",
                    type=str,
                    help="path of the file with the entities to convert (only used by baselines)")

parser.add_argument("--mode",
                    type=str,
                    default="sufficient",
                    choices=["sufficient", "necessary"],
                    help="The explanation mode")

parser.add_argument("--relevance_threshold",
                    type=float,
                    default=None,
                    help="The relevance acceptance threshold to use")

prefilters = [TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER]
parser.add_argument('--prefilter',
                    choices=prefilters,
                    default='graph-based',
                    help="Prefilter type in {} to use in pre-filtering".format(prefilters))

parser.add_argument("--prefilter_threshold",
                    type=int,
                    default=20,
                    help="The number of promising training facts to keep after prefiltering")

args = parser.parse_args()
seed = 42
torch.backends.cudnn.deterministic = True
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())

hyperparameters = {DIMENSION: args.dimension,
                   MARGIN: args.margin,
                   NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
                   REGULARIZER_WEIGHT: args.regularizer_weight,
                   BATCH_SIZE: args.batch_size,
                   LEARNING_RATE: args.learning_rate,
                   EPOCHS: args.max_epochs}

relevance_threshold = args.relevance_threshold
prefilter = args.prefilter

########## LOAD DATASET

# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Reading facts to explain...")
with open(args.facts_to_explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

model = TransE(dataset=dataset, hyperparameters=hyperparameters, init_random=True)  # type: TransE
model.to('cuda')
model.load_state_dict(torch.load(args.model_path))
model.eval()

start_time = time.time()

if args.baseline is None:
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                    relevance_threshold=relevance_threshold)
elif args.baseline == "data_poisoning":
    kelpie = DataPoisoning(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter)
elif args.baseline == "criage":
    kelpie = Criage(model=model, dataset=dataset, hyperparameters=hyperparameters)
elif args.baseline == "k1":
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                    relevance_threshold=relevance_threshold, max_explanation_length=1)
else:
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                    relevance_threshold=relevance_threshold)

testing_fact_2_entities_to_convert = None
if args.mode == "sufficient" and args.entities_to_convert is not None:
    print("Reading entities to convert...")
    testing_fact_2_entities_to_convert = {}
    with open(args.entities_to_convert, "r") as entities_to_convert_file:
        entities_to_convert_lines = entities_to_convert_file.readlines()
        i = 0
        while i < len(entities_to_convert_lines):
            cur_head, cur_rel, cur_name = entities_to_convert_lines[i].strip().split(";")
            assert [cur_head, cur_rel, cur_name] in testing_facts
            cur_entities_to_convert = entities_to_convert_lines[i + 1].strip().split(",")
            testing_fact_2_entities_to_convert[(cur_head, cur_rel, cur_name)] = cur_entities_to_convert
            i += 3

output_lines = []
for i, fact in enumerate(testing_facts):
    head, relation, tail = fact
    print("Explaining fact " + str(i) + " on " + str(
        len(testing_facts)) + ": <" + head + "," + relation + "," + tail + ">")
    head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
                                    dataset.get_id_for_relation_name(relation), \
                                    dataset.get_id_for_entity_name(tail)
    sample_to_explain = (head_id, relation_id, tail_id)

    if args.mode == "sufficient":
        entities_to_convert_ids = None if testing_fact_2_entities_to_convert is None \
            else [dataset.entity_name_2_id[x] for x in testing_fact_2_entities_to_convert[(head, relation, tail)]]

        rule_samples_with_relevance, \
        entities_to_convert_ids = kelpie.explain_sufficient(sample_to_explain=sample_to_explain,
                                                            perspective="head",
                                                            num_promising_samples=args.prefilter_threshold,
                                                            num_entities_to_convert=args.coverage,
                                                            entities_to_convert=entities_to_convert_ids)

        if entities_to_convert_ids is None or len(entities_to_convert_ids) == 0:
            continue
        entities_to_convert = [dataset.entity_id_2_name[x] for x in entities_to_convert_ids]

        rule_facts_with_relevance = []
        for cur_rule_with_relevance in rule_samples_with_relevance:
            cur_rule_samples, cur_relevance = cur_rule_with_relevance

            cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
            cur_rule_facts = ";".join([";".join(x) for x in cur_rule_facts])
            rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))

        print(";".join(fact))
        print(", ".join(entities_to_convert))
        print(", ".join(rule_facts_with_relevance))
        print()
        output_lines.append(";".join(fact) + "\n")
        output_lines.append(",".join(entities_to_convert) + "\n")
        output_lines.append(",".join(rule_facts_with_relevance) + "\n")
        output_lines.append("\n")

    elif args.mode == "necessary":
        rule_samples_with_relevance = kelpie.explain_necessary(sample_to_explain=sample_to_explain,
                                                               perspective="head",
                                                               num_promising_samples=args.prefilter_threshold)
        rule_facts_with_relevance = []
        for cur_rule_with_relevance in rule_samples_with_relevance:
            cur_rule_samples, cur_relevance = cur_rule_with_relevance

            cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
            cur_rule_facts = ";".join([";".join(x) for x in cur_rule_facts])
            rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))
        print(";".join(fact))
        print(", ".join(rule_facts_with_relevance))
        print()
        output_lines.append(";".join(fact) + "\n")
        output_lines.append(",".join(rule_facts_with_relevance) + "\n")
        output_lines.append("\n")

end_time = time.time()
print("Required time: " + str(end_time - start_time) + " seconds")
with open("output.txt", "w") as output:
    output.writelines(output_lines)

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

import argparse
import random
import time

import numpy
import torch

from dataset import ALL_DATASET_NAMES, Dataset
from kelpie import Kelpie as Kelpie
from data_poisoning import DataPoisoning
from criage import Criage
from link_prediction.models.complex import ComplEx
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, REGULARIZER_WEIGHT, EPOCHS, \
    BATCH_SIZE, REGULARIZER_NAME

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument('--dataset',
                    choices=datasets,
                    help="Dataset in {}".format(datasets),
                    required=True)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adagrad',
                    help="Optimizer in {} to use in post-training".format(optimizers))

parser.add_argument('--batch_size',
                    default=100,
                    type=int,
                    help="Batch size to use in post-training")

parser.add_argument('--max_epochs',
                    default=200,
                    type=int,
                    help="Number of epochs to run in post-training")

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Factorization rank.")

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate")

parser.add_argument('--reg',
                    default=0,
                    type=float,
                    help="Regularization weight")

parser.add_argument('--init',
                    default=1e-3,
                    type=float,
                    help="Initial scale")

parser.add_argument('--decay1',
                    default=0.9,
                    type=float,
                    help="Decay rate for the first moment estimate in Adam")

parser.add_argument('--decay2',
                    default=0.999,
                    type=float,
                    help="Decay rate for second moment estimate in Adam")

parser.add_argument('--model_path',
                    help="Path to the model to explain the predictions of",
                    required=True)

parser.add_argument("--facts_to_explain_path",
                    type=str,
                    required=True,
                    help="path of the file with the facts to explain the predictions of.")

parser.add_argument("--coverage",
                    type=int,
                    default=10,
                    help="Number of random entities to extract and convert")

parser.add_argument("--baseline",
                    type=str,
                    default=None,
                    choices=[None, "data_poisoning", "criage"],
                    help="attribute to use when we want to use a baseline rather than the Kelpie engine")

parser.add_argument("--entities_to_convert",
                    type=str,
                    help="path of the file with the entities to convert (only used by baselines)")

parser.add_argument("--mode",
                    type=str,
                    default="sufficient",
                    choices=["sufficient", "necessary"],
                    help="The explanation mode")

args = parser.parse_args()

########## LOAD DATASET

# deterministic!
seed = 42
torch.backends.cudnn.deterministic = True
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())

hyperparameters = {DIMENSION: args.dimension,
                   INIT_SCALE: args.init,
                   LEARNING_RATE: args.learning_rate,
                   OPTIMIZER_NAME: args.optimizer,
                   DECAY_1: args.decay1,
                   DECAY_2: args.decay2,
                   REGULARIZER_WEIGHT: args.reg,
                   EPOCHS: args.max_epochs,
                   BATCH_SIZE: args.batch_size,
                   REGULARIZER_NAME: "N3"}

# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Reading facts to explain...")
with open(args.facts_to_explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True) # type: ComplEx
model.to('cuda')
model.load_state_dict(torch.load(args.model_path))
model.eval()

start_time = time.time()

if args.baseline is None:
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters)
elif args.baseline == "data_poisoning":
    kelpie = DataPoisoning(model=model, dataset=dataset, hyperparameters=hyperparameters)
elif args.baseline == "criage":
    kelpie = Criage(model=model, dataset=dataset, hyperparameters=hyperparameters)
else:
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters)


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
    print("Explaining fact " + str(i) + " on " + str(len(testing_facts)) + ": <" + head + "," + relation + "," + tail + ">")
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
                                                            num_promising_samples=20,
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
                                                               num_promising_samples=20)
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
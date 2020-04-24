"""
This module does explanation for ComplEx model
"""
import argparse
import torch
import numpy

from kelpie.crosse_explanation.dataset import PathDataset
from kelpie.dataset import ALL_DATASET_NAMES
from kelpie.models.complex.model import ComplEx, KelpieComplEx
from kelpie.models.complex.dataset import ComplExDataset, KelpieComplExDataset
from kelpie.models.complex.evaluators import ComplExEvaluator, KelpieComplExEvaluator
from kelpie.models.complex.optimizer import KelpieComplExOptimizer

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="CrossE model-agnostic tool for explaining link predictions")

parser.add_argument('--dataset',
                    choices=datasets,
                    help="Dataset in {}".format(datasets),
                    required=True)

parser.add_argument('--model_path',
                    help="Path to the model to explain the predictions of",
                    required=True)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adagrad',
                    help="Optimizer in {} to use in post-training".format(optimizers))

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Factorization rank.")

parser.add_argument('--head',
                    help="Textual name of the head entity of the test fact to explain",
                    required=True)

parser.add_argument('--relation',
                    help="Textual name of the relation of the test fact to explain",
                    required=True)

parser.add_argument('--tail',
                    help="Textual name of the tail entity of the test fact to explain",
                    required=True)

parser.add_argument('--init',
                    default=1e-3,
                    type=float,
                    help="Initial scale"
)


#   E.G.    explain  why  /m/02mjmr (Barack Obama)  is predicted as the head for
#   /m/02mjmr (Barack Obama) 	/people/person/ethnicity	/m/033tf_ (Irish American)
args = parser.parse_args()


#############  LOAD DATASET

# get the fact to explain and its perspective entity
head, relation, tail = args.head, args.relation, args.tail
entity_to_explain = head if args.perspective.lower() == "head" else tail

# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
original_dataset = ComplExDataset(name=args.dataset,
                                  separator="\t",
                                  load=True)

# get the ids of the elements of the fact to explain and the perspective entity
head_id, relation_id, tail_id = original_dataset.get_id_for_entity_name(head), \
                                original_dataset.get_id_for_relation_name(relation), \
                                original_dataset.get_id_for_entity_name(tail)
original_entity_id = head_id if args.perspective == "head" else tail_id

# create the fact to explain as a numpy array of its ids
original_sample_tuple = (head_id, relation_id, tail_id)
original_sample = numpy.array(original_sample_tuple)

# check that the fact to explain is actually a test fact
assert(original_sample in original_dataset.test_samples)


#############   INITIALIZE MODELS AND THEIR STRUCTURES
print("Loading model at location %s..." % args.model_path)
# instantiate and load the original model from filesystem
original_model = ComplEx(dataset=original_dataset, dimension=args.dimension, init_random=True, init_size=args.init)
original_model.load_state_dict(torch.load(args.model_path))


print("Loading paths dataset %s..." % args.dataset)
explaination_dataset = PathDataset(name=args.dataset,
                                  separator="\t",
                                  load=True)


import argparse

import numpy
import torch

from data_poisoning.data_poisoning_relevance import compute_fact_relevance
from dataset import ALL_DATASET_NAMES, Dataset
from link_prediction.models.complex import ComplEx
from model import DIMENSION, INIT_SCALE

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

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

parser.add_argument('--perspective',
                    choices=['head', 'tail'],
                    default='head',
                    help="Explanation perspective in {}".format(['head', 'tail']))

parser.add_argument('--reg',
                    default=0,
                    type=float,
                    help="Regularization weight"
)

parser.add_argument('--head',
                    type=str,
                    help="Name of the head entity of the fact to explain"
)

parser.add_argument('--relation',
                    type=str,
                    help="Name of the relation of the fact to explain"
)

parser.add_argument('--tail',
                    type=str,
                    help="Name of the tail entity of the fact to explain"
)

parser.add_argument('--init_scale',
                    default=1e-3,
                    type=float,
                    help="Initial scale"
)

parser.add_argument('--decay1',
                    default=0.9,
                    type=float,
                    help="Decay rate for the first moment estimate in Adam"
)
parser.add_argument('--decay2',
                    default=0.999,
                    type=float,
                    help="Decay rate for second moment estimate in Adam"
)

#   E.G.    explain  why  /m/02mjmr (Barack Obama)  is predicted as the head for
#   /m/02mjmr (Barack Obama) 	/people/person/ethnicity	/m/033tf_ (Irish American)
#   or
#   /m/02mjmr (Barack Obama)	/people/person/places_lived./people/place_lived/location	/m/02hrh0_ (Honolulu)
args = parser.parse_args()

# load the dataset and its training samples
original_dataset = Dataset(name=args.dataset, separator="\t", load=True)

# get the ids of the elements of the fact to explain and the perspective entity
head_id, relation_id, tail_id = original_dataset.get_id_for_entity_name(args.head), \
                                original_dataset.get_id_for_relation_name(args.relation), \
                                original_dataset.get_id_for_entity_name(args.tail)

original_entity_id = head_id # if args.perspective == "head" else tail_id

# create the fact to explain as a numpy array of its ids
original_sample = numpy.array((head_id, relation_id, tail_id))

#############   INITIALIZE MODELS AND THEIR STRUCTURES
# instantiate and load the original model from filesystem
original_model = ComplEx(dataset=original_dataset,
                         hyperparameters={DIMENSION: args.dimension, INIT_SCALE:args.init_scale},
                         init_random=True) # type: ComplEx
original_model.load_state_dict(torch.load(args.model_path))
original_model.to('cuda')

sample_relevance_couples = compute_fact_relevance(model=original_model,
                           dataset=original_dataset,
                           perspective="head",
                           sample_to_explain=original_sample,
                           perturbation_step=0.05)

print("10 most relevant triples:")
for training_sample, value in sample_relevance_couples[:10]:
    h, r, t = training_sample
    print(str(original_dataset.entity_id_2_name[h]) + " " + str(original_dataset.relation_id_2_name[r]) + " " + str(
        original_dataset.entity_id_2_name[t]) + ": " + str(value))
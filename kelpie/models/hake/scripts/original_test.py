import argparse
import os
import torch
from kelpie.dataset import ALL_DATASET_NAMES, Dataset

from kelpie.models.hake.model import Hake

# like the test.py script, except that we then use a reimplementation of HAKE's original way of evaluating the model
# this is for testing purposes only.

MODEL_HOME = os.path.abspath("./models/")
ALL_MODEL_NAMES = ["Hake"]

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument('--dataset',
                    choices=ALL_DATASET_NAMES,
                    help="Dataset in {}".format(ALL_DATASET_NAMES)
)

parser.add_argument('--model',
                    choices=ALL_MODEL_NAMES,
                    help="Model in {}".format(ALL_MODEL_NAMES)
)

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Embedding dimension"
)

parser.add_argument('--load',
                    help="path to the model to load",
                    required=False)

# Hake-specific args:

parser.add_argument('--gamma',
                    default=1.0,
                    type=float,
                    help="Gamma"
)

parser.add_argument('--modulus_weight',
                    default=1.0,
                    type=float,
                    help="Modulus weight"
)

parser.add_argument('--phase_weight',
                    default=0.5,
                    type=float,
                    help="Phase weight"
)

parser.add_argument('--cpu_num',
                    default=10,
                    type=int,
                    help="Number of (virtual) CPU cores to use"
)

parser.add_argument('--batch_size',
                    default=1024,
                    type=int,
                    help="Number of samples in each mini-batch in SGD, Adagrad and Adam optimization"
)

parser.add_argument('--test_batch_size',
                    default=4,
                    type=int,
                    help="Number of samples in each mini-batch in SGD, Adagrad and Adam optimization during evaluation"
)

#

args = parser.parse_args()

model_path = "./models/" + "_".join([args.model, args.dataset]) + ".pt"
if args.load is not None:
    model_path = args.load

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Initializing model...")
model = Hake(dataset=dataset, hidden_dim=args.dimension, batch_size=args.batch_size, test_batch_size=args.test_batch_size,
             cpu_num=args.cpu_num, gamma=args.gamma, modulus_weight=args.modulus_weight, phase_weight=args.phase_weight)

model.to('cuda')
model.load_state_dict(torch.load(model_path))
model.eval()

print("\nEvaluating model...")
model.test_step(dataset.test_samples)

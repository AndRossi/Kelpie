import argparse
import os
import torch
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import Evaluator

# todo: when we add more models, we should move these variables to another location
from kelpie.models.hake.model import Hake

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

#

args = parser.parse_args()

model_path = "./models/" + "_".join([args.model, args.dataset]) + ".pt"
if args.load is not None:
    model_path = args.load

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Initializing model...")
model = Hake(dataset=dataset, hidden_dim=args.dimension, gamma=args.gamma, modulus_weight=args.modulus_weight, phase_weight=args.phase_weight)   # type: Hake
model.to('cuda')
model.load_state_dict(torch.load(model_path))
model.eval()

print("\nEvaluating model...")
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=True)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)

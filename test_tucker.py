import argparse
import os
import torch
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import Evaluator
from kelpie.models.tucker.model import TuckER

# todo: when we add more models, we should move these variables to another location
MODEL_HOME = os.path.abspath("./models/")
ALL_MODEL_NAMES = ["TuckER"]

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

parser.add_argument('--entity_dimension',
                    default=1000,
                    type=int,
                    help="Entity embedding dimension"
)

parser.add_argument('--relation_dimension',
                    default=1000,
                    type=int,
                    help="Relation embedding dimension"
)

parser.add_argument('--load',
                    help="path to the model to load")

args = parser.parse_args()

model_path = "./models/" + "_".join([args.model, args.dataset]) + ".pt"
if args.load is not None:
    model_path = args.load

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Initializing model...")
model = TuckER(dataset=dataset,
               entity_dimension=args.entity_dimension,
               relation_dimension=args.relation_dimension,
               input_dropout=0,
               hidden_dropout1=0,
               hidden_dropout2=0,
               init_random=True)
model.to('cuda')
model.load_state_dict(torch.load(model_path))
model.eval()

print("\nEvaluating model...")
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=True)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)

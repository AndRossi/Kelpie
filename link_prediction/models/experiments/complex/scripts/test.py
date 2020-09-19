import argparse
import torch

from dataset import ALL_DATASET_NAMES, Dataset
from link_prediction.evaluation.evaluation import Evaluator
from link_prediction.models.complex import ComplEx
from model import DIMENSION, INIT_SCALE

parser = argparse.ArgumentParser(
    description="Kelpie"
)

parser.add_argument('--dataset',
                    choices=ALL_DATASET_NAMES,
                    help="Dataset in {}".format(ALL_DATASET_NAMES)
)

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Embedding dimension"
)

parser.add_argument('--init_scale',
                    default=1e-3,
                    type=float,
                    help="Initial scale"
)

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate"
)

parser.add_argument('--load',
                    help="path to the model to load",
                    required=True)

args = parser.parse_args()

model_path = args.load

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

hyperparameters = {DIMENSION: args.dimension, INIT_SCALE: args.init_scale}
print("Initializing model...")
model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)   # type: ComplEx
model.to('cuda')
model.load_state_dict(torch.load(model_path))
model.eval()

print("Evaluating model...")
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=True)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)
import argparse
import torch

from dataset import ALL_DATASET_NAMES, Dataset
from evaluation import Evaluator
from models.complex.model import ComplEx

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

parser.add_argument('--init',
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

print("Initializing model...")
model = ComplEx(dataset=dataset, dimension=args.dimension, init_random=True, init_size=args.init)   # type: ComplEx
model.to('cuda')
model.load_state_dict(torch.load(model_path))
model.eval()

print("Evaluating model...")
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=True)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)
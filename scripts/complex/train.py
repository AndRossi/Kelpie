import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

import argparse
import numpy
import torch

from config import MODEL_PATH
from dataset import ALL_DATASET_NAMES, Dataset

from link_prediction.evaluation.evaluation import Evaluator
from link_prediction.models.complex import ComplEx
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.models.model import OPTIMIZER_NAME, LEARNING_RATE, REGULARIZER_NAME, REGULARIZER_WEIGHT, BATCH_SIZE, DECAY_1, DECAY_2, \
    DIMENSION, INIT_SCALE, EPOCHS

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument('--dataset',
                    choices=ALL_DATASET_NAMES,
                    help="Dataset in {}".format(ALL_DATASET_NAMES)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adagrad',
                    help="Optimizer in {}".format(optimizers)
)

parser.add_argument('--max_epochs',
                    default=50,
                    type=int,
                    help="Number of epochs."
)

parser.add_argument('--valid',
                    default=-1,
                    type=float,
                    help="Number of epochs before valid."
)

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Embedding dimension"
)

parser.add_argument('--batch_size',
                    default=1000,
                    type=int,
                    help="Number of samples in each mini-batch in SGD, Adagrad and Adam optimization"
)

parser.add_argument('--reg',
                    default=0,
                    type=float,
                    help="Regularization weight"
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

parser.add_argument('--load',
                    help="path to the model to load",
                    required=False)

args = parser.parse_args()

#deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

if args.load is not None:
    model_path = args.load
else:
    model_path = os.path.join(MODEL_PATH, "_".join(["ComplEx", args.dataset]) + ".pt")
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)


hyperparameters = {DIMENSION:args.dimension,
                   INIT_SCALE:args.init_scale,
                   OPTIMIZER_NAME:args.optimizer,
                   BATCH_SIZE:args.batch_size,
                   EPOCHS:args.max_epochs,
                   LEARNING_RATE:args.learning_rate,
                   DECAY_1:args.decay1,
                   DECAY_2:args.decay2,
                   REGULARIZER_NAME:'N3',
                   REGULARIZER_WEIGHT:args.reg}

print("Initializing model...")
model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)   # type: ComplEx
model.to('cuda')
if args.load is not None:
    model.load_state_dict(torch.load(model_path))

print("Training model...")
optimizer = MultiClassNLLOptimizer(model=model,
                                  hyperparameters=hyperparameters)

optimizer.train(train_samples=dataset.train_samples,
                save_path=model_path,
                evaluate_every=args.valid,
                valid_samples=dataset.valid_samples)

print("Evaluating model...")
mrr, h1, h10, mr = Evaluator(model=model).evaluate(samples=dataset.test_samples, write_output=False)
print("\tTest Hits@1: %f" % h1)
print("\tTest Hits@10: %f" % h10)
print("\tTest Mean Reciprocal Rank: %f" % mrr)
print("\tTest Mean Rank: %f" % mr)


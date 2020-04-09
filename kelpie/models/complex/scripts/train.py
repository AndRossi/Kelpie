import argparse
import os

import torch
from torch import optim

from kelpie.dataset import ALL_DATASET_NAMES
from kelpie.models.complex.complex import ComplEx
from kelpie.models.complex.dataset import ComplExDataset
from kelpie.models.complex.evaluators import ComplExEvaluator
from kelpie.models.complex.optimizers import ComplExOptimizer
from kelpie.models.complex.regularizers import N2, N3

# todo: when we add more models, we should move these variables to another location
MODEL_HOME = os.path.abspath("./models/")
ALL_MODEL_NAMES = ["ComplEx"]

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

regularizers = ['N3', 'N2']
parser.add_argument('--regularizer',
                    choices=regularizers,
                    default='N3',
                    help="Regularizer in {}".format(regularizers)
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
                    default=3,
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
                    help="path to the model to load")

args = parser.parse_args()

model_path = "./models/" + "_".join([args.model, args.dataset]) + ".pt"
if args.load is not None:
    model_path = args.load


regularizer = {
    'N2': N2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]


dataset = ComplExDataset(args.dataset, separator="\t")
print("Loading %s dataset..." % dataset.name)
dataset.load()

model = ComplEx(dataset=dataset, dimension=args.dimension, init_random=True, init_size=args.init)
model.to('cuda')

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = ComplExOptimizer(model, regularizer, optim_method, args.batch_size)

if args.load is not None:
    model.load_state_dict(torch.load(model_path))
    model.eval()

evaluator = ComplExEvaluator()
train_samples = dataset.get_train_samples(with_inverse=True)
for e in range(args.max_epochs):

    cur_loss = optimizer.epoch(train_samples)

    if (e + 1) % args.valid == 0:
        mrr, h1 = evaluator.eval(model, dataset.get_valid_samples(with_inverse=False), write_output=False)

        print("\tValidation Hits@1: %f" % h1)
        print("\tValidation Mean Reciprocal Rank: %f" % mrr)

        print("\t saving model...")
        torch.save(model.state_dict(), model_path)
        print("\t done.")

mrr, h1 = evaluator.eval(model, dataset.get_test_samples(with_inverse=False), write_output=False)
print("\n")
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)

import argparse
import os

import torch
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import Evaluator

# todo: when we add more models, we should move these variables to another location
from kelpie.models.hake.data import get_train_iterator_from_dataset
from kelpie.models.hake.model import Hake
from kelpie.models.hake.optimizer import HakeOptimizer

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

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adam',
                    help="Optimizer in {}".format(optimizers)
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

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Embedding dimension"
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

parser.add_argument('--learning_rate',
                    default=0.0001,
                    type=float,
                    help="Learning rate"
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

parser.add_argument('--no_decay',
                    default=False,
                    type=bool,
                    help="Disables decay if true"
)

parser.add_argument('--adversarial_temperature',
                    default=1.0,
                    type=float,
                    help="Adversarial temperature"
)

parser.add_argument('--init_step',
                    default=0,
                    type=int,
                    help="Initial training step number"
)

parser.add_argument('--cpu_num',
                    default=10,
                    type=int,
                    help="Number of (virtual) CPU cores to use"
)

parser.add_argument('--negative_sample_size',
                    default=128,
                    type=int,
                    help = "Number of negative samples"
)

#

args = parser.parse_args()

model_path = "./models/" + "_".join(["Hake", args.dataset]) + ".pt"
if args.load is not None:
    model_path = args.load

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Initializing model...")
model = Hake(dataset=dataset, hidden_dim=args.dimension, batch_size=args.batch_size, test_batch_size=args.test_batch_size,
             cpu_num=args.cpu_num, gamma=args.gamma, modulus_weight=args.modulus_weight, phase_weight=args.phase_weight)
model.to('cuda')
if args.load is not None:
    model.load_state_dict(torch.load(model_path))

print("Training model...")
optimizer = HakeOptimizer(model=model,
                             optimizer_name=args.optimizer,
                             learning_rate=args.learning_rate,
                             no_decay=args.no_decay,
                             max_steps=args.max_epochs,
                             adversarial_temperature=args.adversarial_temperature,
                             save_path=model_path)

train_iterator = get_train_iterator_from_dataset(triples=dataset.train_samples,
                                    num_entities=dataset.num_entities,
                                    num_relations=dataset.num_relations,
                                    cpu_num=args.cpu_num,
                                    batch_size=args.batch_size,
                                    negative_sample_size=args.negative_sample_size
)

optimizer.train(train_iterator=train_iterator,
                init_step=args.init_step,
                evaluate_every=args.valid,
                valid_samples=dataset.valid_samples)

print("\nEvaluating model...")
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=False)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)

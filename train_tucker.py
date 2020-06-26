import argparse
import os

import torch
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import Evaluator
from kelpie.models.tucker.model import TuckER
from kelpie.models.tucker.optimizer import TuckEROptimizer

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

optimizers = ['Adam']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adam',
                    help="Optimizer in {}".format(optimizers)
)

schedulers = ['ExponentialLR']
parser.add_argument('--scheduler',
                    choices=schedulers,
                    default='ExponentialLR',
                    help="Scheduler in {}".format(schedulers)
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

parser.add_argument('--batch_size',
                    default=1000,
                    type=int,
                    help="Number of samples in each mini-batch in SGD, Adagrad and Adam optimization"
)

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate"
)

parser.add_argument('--decay',
                    default=1.0,
                    type=float,
                    help="Decay rate"
)

parser.add_argument('--input_dropout',
                    default=0.3,
                    type=float,
                    help="Input layer dropout"
)

parser.add_argument('--hidden_dropout1',
                    default=0.4,
                    type=float,
                    help="Dropout after the first hidden layer"
)

parser.add_argument('--hidden_dropout2',
                    default=0.5,
                    type=float,
                    help="Dropout after the second hidden layer"
)

parser.add_argument('--label_smoothing',
                    default=0.1,
                    type=float,
                    help="Amount of label smoothing"
)

parser.add_argument('--load',
                    help="path to the model to load",
                    required=False)

args = parser.parse_args()

model_path = "./models/" + "_".join(["TuckER", args.dataset]) + ".pt"
if args.load is not None:
    model_path = args.load

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Initializing model...")
model = TuckER(dataset=dataset,
               entity_dimension=args.entity_dimension,
               relation_dimension=args.relation_dimension,
               input_dropout=args.input_dropout,
               hidden_dropout1=args.hidden_dropout1,
               hidden_dropout2=args.hidden_dropout2,
               init_random=True)   # type: TuckER
model.to('cuda')
if args.load is not None:
    model.load_state_dict(torch.load(model_path))

print("Training model...")
optimizer = TuckEROptimizer(model=model,
                             optimizer_name=args.optimizer,
                             scheduler_name=args.scheduler,
                             batch_size=args.batch_size,
                             learning_rate=args.learning_rate,
                             decay=args.decay,
                             label_smoothing=args.label_smoothing)
optimizer.train(train_samples=dataset.train_samples,
                max_epochs=args.max_epochs,
                save_path=model_path,
                evaluate_every=args.valid,
                valid_samples=dataset.valid_samples)

print("\nEvaluating model...")
model.cpu().eval()
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=False)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)

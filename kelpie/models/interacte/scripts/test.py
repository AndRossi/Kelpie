import argparse
import os
import torch
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import Evaluator
from kelpie.models.interacte.model import InteractE

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

optimizers = ['Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adam',
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

parser.add_argument('--embed_dim',
                    default=1000,
                    type=int,
                    help="Embedding dimension"
)

parser.add_argument('--batch_size',
                    default=1000,
                    type=int,
                    help="Number of samples in each mini-batch in SGD, Adagrad and Adam optimization"
)

parser.add_argument('--weight_decay',
                    default=0,
                    type=float,
                    help="Regularization weight"
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
                    help="path to the model to load"
)

parser.add_argument('--inp_drop_p',
                    default=0.5,
                    type=float,
                    help="Dropout regularization probability for the input embeddings"
)

parser.add_argument('--hid_drop_p',
                    default=0.5,
                    type=float,
                    help="Dropout regularization probability for the hidden layer"
)

parser.add_argument('--feat_drop_p',
                    default=0.5,
                    type=float,
                    help="Dropout regularization probability for the feature matrix"
)

parser.add_argument('--kernel_size',
                    default=9,
                    type=int,
                    help="Size of the kernel function window"
)

parser.add_argument('--num_filt_conv',
                    default=96,
                    type=int,
                    help="Number of convolution filters"
)

parser.add_argument('--strategy',
                    default='one_to_n',
                    help="Choose the strategy: one_to_n"
)

args = parser.parse_args()

model_path = "./models/" + "_".join([args.model, args.dataset]) + ".pt"
if args.load is not None:
    model_path = args.load

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Initializing model...")
model = InteractE(dataset=dataset,
                  embed_dim = args.embed_dim, 
                  k_h = 20,
                  k_w = 10,
                  inp_drop_p = args.inp_drop_p,
                  hid_drop_p = args.hid_drop_p,
                  feat_drop_p = args.feat_drop_p,
                  num_perm = args.num_perm,
                  kernel_size = args.kernel_size,
                  num_filt_conv = args.num_filt_conv,
                  strategy = args.strategy
)

model.to('cuda')
model.load_state_dict(torch.load(model_path))
model.eval()

print("\nEvaluating model...")
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=True)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)
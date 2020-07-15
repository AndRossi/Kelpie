from collections import defaultdict

import torch
from torch.optim.lr_scheduler import ExponentialLR
import argparse

from kelpie.dataset import Dataset
from kelpie.evaluation import Evaluator
from kelpie.models.tucker.model import TuckER
from kelpie.models.tucker.optimizer import TuckEROptimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--decay_rate", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--entity_dimension", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--relation_dimension", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument('--test', action="store_true", default=False,
                        help="Run test instead of training.")

    args = parser.parse_args()
    #torch.backends.cudnn.deterministic = True
    #seed = 20
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #if torch.cuda.is_available:
    #    torch.cuda.manual_seed_all(seed)

    dataset_name = args.dataset
    dataset = Dataset(dataset_name)

    tucker = TuckER(dataset=dataset, entity_dimension=args.entity_dimension, relation_dimension=args.relation_dimension,
                   input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1,
                   hidden_dropout2=args.hidden_dropout2, init_random=True)

    optimizer = TuckEROptimizer(model=tucker, batch_size=args.batch_size, learning_rate=args.lr,
                                decay=args.decay_rate, label_smoothing=args.label_smoothing)

    optimizer.train(train_samples=dataset.train_samples, max_epochs=args.num_iterations, evaluate_every=10,
                    save_path="./stored_models/TuckER" + dataset_name + ".pt",
                    valid_samples=dataset.valid_samples)
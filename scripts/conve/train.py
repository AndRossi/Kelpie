import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

import argparse
import numpy
import torch

from config import MODEL_PATH
from dataset import Dataset, ALL_DATASET_NAMES
from link_prediction.evaluation.evaluation import Evaluator
from link_prediction.models.conve import ConvE
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.models.model import INPUT_DROPOUT, BATCH_SIZE, LEARNING_RATE, DECAY, LABEL_SMOOTHING, \
    EPOCHS, DIMENSION, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, HIDDEN_LAYER_SIZE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        type=str,
                        choices=ALL_DATASET_NAMES,
                        help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")

    parser.add_argument("--max_epochs",
                        type=int,
                        default=1000,
                        help="Number of epochs.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.0005,
                        help="Learning rate.")

    parser.add_argument("--decay_rate",
                        type=float,
                        default=1.0,
                        help="Decay rate.")

    parser.add_argument("--dimension",
                        type=int,
                        default=200,
                        help="Embedding dimensionality.")

    parser.add_argument("--valid",
                        type=int,
                        default=-1,
                        help="Validate after a cycle of x epochs")

    parser.add_argument("--input_dropout",
                        type=float,
                        default=0.3,
                        nargs="?",
                        help="Input layer dropout.")

    parser.add_argument("--hidden_dropout",
                        type=float,
                        default=0.4,
                        help="Dropout after the hidden layer.")

    parser.add_argument("--feature_map_dropout",
                        type=float,
                        default=0.5,
                        help="Dropout after the convolutional layer.")

    parser.add_argument("--label_smoothing",
                        type=float,
                        default=0.1,
                        help="Amount of label smoothing.")

    parser.add_argument('--hidden_size',
                        type=int,
                        default=9728,
                        help='The side of the hidden layer. '
                             'The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    seed = 42
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    dataset_name = args.dataset
    dataset = Dataset(dataset_name)

    hyperparameters = {DIMENSION: args.dimension,
                       INPUT_DROPOUT: args.input_dropout,
                       FEATURE_MAP_DROPOUT: args.feature_map_dropout,
                       HIDDEN_DROPOUT: args.hidden_dropout,
                       HIDDEN_LAYER_SIZE: args.hidden_size,
                       BATCH_SIZE: args.batch_size,
                       LEARNING_RATE: args.learning_rate,
                       DECAY: args.decay_rate,
                       LABEL_SMOOTHING: args.label_smoothing,
                       EPOCHS: args.max_epochs}

    conve = ConvE(dataset=dataset, hyperparameters=hyperparameters, init_random=True) # type: ConvE

    optimizer = BCEOptimizer(model=conve, hyperparameters=hyperparameters)

    optimizer.train(train_samples=dataset.train_samples, evaluate_every=10,
                    save_path=os.path.join(MODEL_PATH, "ConvE_" + dataset_name + ".pt"),
                    valid_samples=dataset.valid_samples)

    print("Evaluating model...")
    mrr, h1, h10, mr = Evaluator(model=conve).evaluate(samples=dataset.test_samples, write_output=False)
    print("\tTest Hits@1: %f" % h1)
    print("\tTest Hits@10: %f" % h10)
    print("\tTest Mean Reciprocal Rank: %f" % mrr)
    print("\tTest Mean Rank: %f" % mr)

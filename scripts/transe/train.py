import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

import argparse
import random

import numpy
import torch

from config import MODEL_PATH
from dataset import Dataset, ALL_DATASET_NAMES
from link_prediction.evaluation.evaluation import Evaluator
from link_prediction.models.transe import TransE
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from link_prediction.models.model import BATCH_SIZE, LEARNING_RATE, EPOCHS, DIMENSION, MARGIN, NEGATIVE_SAMPLES_RATIO, REGULARIZER_WEIGHT

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

    parser.add_argument("--dimension",
                        type=int,
                        default=200,
                        help="Embedding dimensionality.")

    parser.add_argument("--margin",
                        type=int,
                        default=5,
                        help="Margin for pairwise ranking loss.")

    parser.add_argument("--negative_samples_ratio",
                        type=int,
                        default=3,
                        help="Number of negative samples for each positive sample.")

    parser.add_argument("--regularizer_weight",
                        type=float,
                        default=0.0,
                        help="Weight for L2 regularization.")

    parser.add_argument("--valid",
                        type=int,
                        default=-1,
                        help="Validate after a cycle of x epochs")

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    seed = 42
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    dataset_name = args.dataset
    dataset = Dataset(dataset_name)

    hyperparameters = {DIMENSION: args.dimension,
                       MARGIN: args.margin,
                       NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
                       REGULARIZER_WEIGHT: args.regularizer_weight,
                       BATCH_SIZE: args.batch_size,
                       LEARNING_RATE: args.learning_rate,
                       EPOCHS: args.max_epochs}

    transe = TransE(dataset=dataset, hyperparameters=hyperparameters, init_random=True) # type: TransE

    optimizer = PairwiseRankingOptimizer(model=transe, hyperparameters=hyperparameters)

    optimizer.train(train_samples=dataset.train_samples, evaluate_every=args.valid,
                    save_path=os.path.join(MODEL_PATH, "TransE_" + dataset_name + ".pt"),
                    valid_samples=dataset.valid_samples)

    print("Evaluating model...")
    mrr, h1, h10, mr = Evaluator(model=transe).evaluate(samples=dataset.test_samples, write_output=False)
    print("\tTest Hits@1: %f" % h1)
    print("\tTest Hits@10: %f" % h10)
    print("\tTest Mean Reciprocal Rank: %f" % mrr)
    print("\tTest Mean Rank: %f" % mr)

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
                    help="Which dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10.")
    parser.add_argument("--entity_dimension", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--relation_dimension", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--model_path", type=str, default=None,
                    help="Path where to find the model to evaluate in the filesystem.")

    args = parser.parse_args()
    #torch.backends.cudnn.deterministic = True
    #seed = 20
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #if torch.cuda.is_available:
    #    torch.cuda.manual_seed_all(seed)

    print("Loading %s dataset..." % args.dataset)
    dataset = Dataset(name=args.dataset, separator="\t", load=True)

    print("Initializing model...")
    tucker = TuckER(dataset=dataset,
                    entity_dimension=args.entity_dimension,
                    relation_dimension=args.relation_dimension,
                    init_random=True) # type: TuckER

    model_path = args.model_path if args.model_path is not None else "./models/TuckER_" + args.dataset + ".pt"
    tucker.load_state_dict(torch.load(model_path))
    tucker.eval()

    print("\nEvaluating model...")
    mrr, h1 = Evaluator(model=tucker).eval(samples=dataset.valid_samples, write_output=True)
    print("\tTest Hits@1: %f" % h1)
    print("\tTest Mean Reciprocal Rank: %f" % mrr)
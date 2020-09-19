import argparse
import os

from config import MODEL_PATH
from dataset import Dataset, ALL_DATASET_NAMES
from evaluation import Evaluator
from models.tucker.model import TuckER
from optimization.bce_optimizer import BCEOptimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        type=str,
                        choices=ALL_DATASET_NAMES,
                        help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")

    parser.add_argument("--max_epochs",
                        type=int, default=500,
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

    parser.add_argument("--entity_dimension",
                        type=int,
                        default=200,
                        help="Entity embedding dimensionality.")

    parser.add_argument("--relation_dimension",
                        type=int,
                        default=200,
                        help="Relation embedding dimensionality.")

    parser.add_argument("--valid",
                        type=int,
                        default=-1,
                        help="Validate after a cycle of x epochs")

    parser.add_argument("--input_dropout",
                        type=float,
                        default=0.3,
                        nargs="?",
                        help="Input layer dropout.")

    parser.add_argument("--hidden_dropout_1",
                        type=float,
                        default=0.4,
                        help="Dropout after the first hidden layer.")

    parser.add_argument("--hidden_dropout_2",
                        type=float,
                        default=0.5,
                        help="Dropout after the second hidden layer.")

    parser.add_argument("--label_smoothing",
                        type=float,
                        default=0.1,
                        help="Amount of label smoothing.")

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
                   input_dropout=args.input_dropout, hidden_dropout_1=args.hidden_dropout_1,
                   hidden_dropout_2=args.hidden_dropout_2, init_random=True) # type: TuckER

    optimizer = BCEOptimizer(model=tucker, batch_size=args.batch_size, learning_rate=args.learning_rate,
                             decay=args.decay_rate, label_smoothing=args.label_smoothing)

    optimizer.train(train_samples=dataset.train_samples, max_epochs=args.max_epochs, evaluate_every=10,
                    save_path=os.path.join(MODEL_PATH, "TuckER_" + dataset_name + ".pt"),
                    valid_samples=dataset.valid_samples)

    print("Evaluating model...")
    mrr, h1 = Evaluator(model=tucker).eval(samples=dataset.test_samples, write_output=False)
    print("\tTest Hits@1: %f" % h1)
    print("\tTest Mean Reciprocal Rank: %f" % mrr)

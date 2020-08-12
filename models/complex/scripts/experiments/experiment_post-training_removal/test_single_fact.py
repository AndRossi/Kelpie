import argparse
import os
import torch
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import Evaluator
from kelpie.models.complex.model import ComplEx

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

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Embedding dimension"
)

parser.add_argument('--init',
                    default=1e-3,
                    type=float,
                    help="Initial scale"
)

parser.add_argument('--load',
                    help="path to the model to load",
                    required=True)

args = parser.parse_args()


print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Initializing model...")
model = ComplEx(dataset=dataset, dimension=args.dimension, init_random=True, init_size=args.init) # type: ComplEx
model.to('cuda')
model.load_state_dict(torch.load(args.load))
model.eval()

print("\nEvaluating model...")

head, relation, tail = "/m/0174qm", "/location/location/containedby", "/m/02jx1"
head_id, relation_id, tail_id = dataset.entity_name_2_id[head], dataset.relation_name_2_id[relation], dataset.entity_name_2_id[tail]
(direct_score, inverse_score), \
(head_rank, tail_rank), \
(head_predictions, tail_predictions) = model.predict_sample((head_id, relation_id, tail_id))


mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=True)

print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)
print("\tResults on test fact <%s, %s, %s>:" % (head, relation, tail))
print("\t\tDirect score: %f; Inverse score: %f" % (direct_score, inverse_score))
print("\t\tHead rank: %f; Tail rank: %f" % (head_rank, tail_rank))
print("\t\tHead predictions: [" + ';'.join([dataset.entity_id_2_name[x] for x in head_predictions[:head_rank]]) + "]")
print("\t\tTail predictions: [" + ';'.join([dataset.entity_id_2_name[x] for x in tail_predictions[:tail_rank]]) + "]")
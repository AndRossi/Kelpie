"""
This module does explanation for ComplEx model
"""
import argparse
from collections import defaultdict

import torch
import numpy

from dataset import ALL_DATASET_NAMES, Dataset
from evaluation import KelpieEvaluator
from dataset import KelpieDataset
from models.complex.model import ComplEx, KelpieComplEx
from optimization.multiclass_nll_optimizer import KelpieMultiClassNLLptimizer

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument('--dataset',
                    choices=datasets,
                    help="Dataset in {}".format(datasets),
                    required=True)

parser.add_argument('--model_path',
                    help="Path to the model to explain the predictions of",
                    required=True)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adagrad',
                    help="Optimizer in {} to use in post-training".format(optimizers))

parser.add_argument('--batch_size',
                    default=100,
                    type=int,
                    help="Batch size to use in post-training")

parser.add_argument('--max_epochs',
                    default=200,
                    type=int,
                    help="Number of epochs to run in post-training")

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Factorization rank.")

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate")

parser.add_argument('--perspective',
                    choices=['head', 'tail'],
                    default='head',
                    help="Explanation perspective in {}".format(['head', 'tail']))

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

#   E.G.    explain  why  /m/02mjmr (Barack Obama)  is predicted as the head for
#   /m/02mjmr (Barack Obama) 	/people/person/ethnicity	/m/033tf_ (Irish American)
#   or
#   /m/02mjmr (Barack Obama)	/people/person/places_lived./people/place_lived/location	/m/02hrh0_ (Honolulu)
args = parser.parse_args()

#deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

#############  LOAD DATASET

# get the fact to explain and its perspective entity

# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
original_dataset = Dataset(name=args.dataset, separator="\t", load=True)

#############   INITIALIZE MODELS AND THEIR STRUCTURES
print("Loading model at location %s..." % args.model_path)
# instantiate and load the original model from filesystem
original_model = ComplEx(dataset=original_dataset,
                         dimension=args.dimension,
                         init_random=True,
                         init_size=args.init) # type: ComplEx
original_model.load_state_dict(torch.load(args.model_path))
original_model.to('cuda')

entity_2_test_degree = defaultdict(lambda:0)
for sample in original_dataset.test_samples:
    entity_2_test_degree[sample[0]] += 1
    entity_2_test_degree[sample[2]] += 1

entity_2_train_degree = defaultdict(lambda:0)
for sample in original_dataset.train_samples:

    if entity_2_test_degree[sample[0]] > 0:
        entity_2_train_degree[sample[0]] += 1

    if entity_2_test_degree[sample[2]] > 0:
        entity_2_train_degree[sample[2]] += 1

entity_degree_couples = sorted(list(entity_2_train_degree.items()), key=lambda x: x[1], reverse=True)

outlines = []

for i in range(len(entity_degree_couples)):
    if i == 0: continue
    if i%3 != 0: continue

    original_entity_id, degree = entity_degree_couples[i]
    kelpie_dataset = KelpieDataset(dataset=original_dataset, entity_id=original_entity_id)

    print("Entity " + kelpie_dataset.entity_id_2_name[original_entity_id] +  "(" +  str(original_entity_id) + ") with degree " + str(degree))

    ############ EXTRACT TEST FACTS AND TRAINING FACTS


    # use model_to_explain to initialize the Kelpie model
    kelpie_model = KelpieComplEx(model=original_model, dataset=kelpie_dataset, init_size=1e-3) # type: KelpieComplEx
    kelpie_model.to('cuda')

    ###########  BUILD THE OPTIMIZER AND RUN POST-TRAINING
    optimizer = KelpieMultiClassNLLptimizer(model=kelpie_model,
                                           optimizer_name=args.optimizer,
                                           batch_size=args.batch_size,
                                           learning_rate=args.learning_rate,
                                           decay1=args.decay1,
                                           decay2=args.decay2,
                                           regularizer_name="N3",
                                           regularizer_weight=args.reg)


    optimizer.train(train_samples=kelpie_dataset.kelpie_train_samples, max_epochs=args.max_epochs)

    ###########  EXTRACT RESULTS

    kelpie_entity_id = kelpie_dataset.kelpie_entity_id

    ### Evaluation on original entity
    # Kelpie model on all facts containing the original entity
    print("Original entity performances (both head and tail predictions):")
    original_mrr, original_h1 = KelpieEvaluator(kelpie_model).eval(samples=kelpie_dataset.original_test_samples, original_mode=True)
    print("\tMRR: %f\n\tH@1: %f" % (original_mrr, original_h1))


    ### Evaluation on kelpie entity
    # results on all facts containing the kelpie entity
    print("Kelpie entity performances (both head and tail predictions):")
    kelpie_mrr, kelpie_h1 = KelpieEvaluator(kelpie_model).eval(samples=kelpie_dataset.kelpie_test_samples, original_mode=False)
    print("\tMRR: %f\n\tH@1: %f" % (kelpie_mrr, kelpie_h1))


    outlines.append("\t".join([kelpie_dataset.entity_id_2_name[original_entity_id],
                               str(original_mrr), str(original_h1), str(kelpie_mrr), str(kelpie_h1)]) + "\n")

    with open("performance_difference_" + original_dataset.name + ".txt", "w") as outfile:
        outfile.writelines(outlines)
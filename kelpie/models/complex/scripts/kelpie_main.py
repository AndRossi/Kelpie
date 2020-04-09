"""
This module does explanation for ComplEx model
"""
import argparse
import torch
from torch import optim
import numpy

from kelpie.dataset import ALL_DATASET_NAMES
from kelpie.models.complex.complex import ComplEx, KelpieComplEx
from kelpie.models.complex.dataset import ComplExDataset, KelpieComplExDataset
from kelpie.models.complex.evaluators import ComplExEvaluator
from kelpie.models.complex.optimizers import KelpieComplExOptimizer
from kelpie.models.complex.regularizers import N3

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

parser.add_argument('--head',
                    help="Textual name of the head entity of the test fact to explain",
                    required=True)

parser.add_argument('--relation',
                    help="Textual name of the relation of the test fact to explain",
                    required=True)

parser.add_argument('--tail',
                    help="Textual name of the tail entity of the test fact to explain",
                    required=True)

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
args = parser.parse_args()


#############  LOAD DATASET

# get the fact to explain and its perspective entity
head, relation, tail = args.head, args.relation, args.tail
entity_to_explain = head if args.perspective.lower() == "head" else tail

# load the dataset and its training samples
original_dataset = ComplExDataset(name=args.dataset, separator="\t")
print("Loading dataset %s..." % args.dataset)
original_dataset.load()

# get the ids of the elements of the fact to explain and the perspective entity
head_id, relation_id, tail_id = original_dataset.get_id_for_entity_name(head), \
                                original_dataset.get_id_for_relation_name(relation), \
                                original_dataset.get_id_for_entity_name(tail)
original_entity_id = head_id if args.perspective == "head" else tail_id

# create the fact to explain as a numpy array of its ids
original_sample_tuple = (head_id, relation_id, tail_id)
original_sample = numpy.array(original_sample_tuple)

# check that the fact to explain is actually a test fact
assert(original_sample in original_dataset.test_samples)



#############   INITIALIZE MODELS AND THEIR STRUCTURES
print("Loading model at location %s..." % args.model_path)
# instantiate and load the original model from filesystem
original_model = ComplEx(dataset=original_dataset, dimension=args.dimension, init_random=True, init_size=args.init)
original_model.load_state_dict(torch.load(args.model_path))
original_model.to('cuda')

print("Wrapping the original model in a Kelpie explainable model...")
# use model_to_explain to initialize the Kelpie model
kelpie_dataset = KelpieComplExDataset(dataset=original_dataset, entity_id=original_entity_id)
kelpie_model = KelpieComplEx(model=original_model, dataset=kelpie_dataset, init_size=1e-3)
kelpie_model.to('cuda')


############ EXTRACT TEST FACTS AND TRAINING FACTS

print("Extracting train and test samples for the original and the kelpie entities...")
# extract all training facts and test facts involving the entity to explain
# and replace the id of the entity to explain with the id of the fake kelpie entity
original_entity_test_samples = kelpie_dataset.get_test_samples(with_inverse=False)
kelpie_test_samples = kelpie_dataset.get_kelpie_test_samples(with_inverse=False)
kelpie_training_facts = kelpie_dataset.get_kelpie_train_samples(with_inverse=True)
# TODO: in time, we should remove with_inverse, and let the model handle inverse facts
# (just as we now do for evaluators, which extracts inverse facts by itself.
# This is not something that kelpie_main.py should know.


# TODO: you should perturbate training facts here; in time, it should probably become a KelpieDataset responsibility


###########  BUILD THE OPTIMIZER AND RUN POST-TRAINING

print("Running post-training on the Kelpie model...")
# build the torch Optimizer object based on the passed name and parameters (learning rate and decays if Adam)
optim_method = {
    'Adagrad': lambda: optim.Adagrad(kelpie_model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(kelpie_model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(kelpie_model.parameters(), lr=args.learning_rate)
}[args.optimizer]()
# build the regularizer with the passed weight
regularizer =  N3(args.reg)
# batch size is the minimum between the passed value and the number of Kelpie training facts
batch_size = min(args.batch_size, len(kelpie_training_facts))
# build the optimizer from the kelpie model, the regularized, the optimization method and the batch size
optimizer = KelpieComplExOptimizer(kelpie_model, regularizer, optim_method, batch_size)

# post-training
cur_loss = 0
for e in range(args.max_epochs):
    cur_loss = optimizer.epoch(kelpie_training_facts)


###########  EXTRACT RESULTS

print("\n\nExtracting results...")
kelpie_entity_id = kelpie_dataset.kelpie_entity_id
kelpie_sample_tuple = (kelpie_entity_id, relation_id, tail_id) if args.perspective == "head" else (head_id, relation_id, kelpie_entity_id)
kelpie_sample = numpy.array(kelpie_sample_tuple)

### Evaluation on original entity

# results on original fact
scores, ranks, predictions = original_model.predict_sample(original_sample)
print("Original model on original test fact: <%s, %s, %s>" % original_sample_tuple)
print("\tDirect fact score: %f; Inverse fact score: %f" % (scores[0], scores[1]))
print("\tHead Rank: %f" % ranks[0])
print("\tTail Rank: %f" % ranks[1])

# results on original fact
scores, ranks, predictions = kelpie_model.predict_sample(original_sample)
print("Kelpie model on original test fact: <%s, %s, %s>" % original_sample_tuple)
print("\tDirect fact score: %f; Inverse fact score: %f" % (scores[0], scores[1]))
print("\tHead Rank: %f" % ranks[0])
print("\tTail Rank: %f" % ranks[1])

# results on all facts containing the original entity
print("\nKelpie model on all test facts containing the original entity:")
mrr, h1 = ComplExEvaluator().eval(kelpie_model, original_entity_test_samples)
print("\tMRR: %f\n\tH@1: %f" % (mrr, h1))


### Evaluation on kelpie entity

# results on kelpie fact
scores, ranks, _ = kelpie_model.predict_sample(kelpie_sample)
print("Kelpie model on original test fact: <%s, %s, %s>" % kelpie_sample_tuple)
print("\tDirect fact score: %f; Inverse fact score: %f" % (scores[0], scores[1]))
print("\tHead Rank: %f" % ranks[0])
print("\tTail Rank: %f" % ranks[1])

# results on all facts containing the kelpie entity
print("\nKelpie model on all test facts containing the Kelpie entity:")
mrr, h1 = ComplExEvaluator().eval(kelpie_model, kelpie_test_samples)
print("\tMRR: %f\n\tH@1: %f" % (mrr, h1))


print("\n\nComputing embedding distances...")
with torch.no_grad():
    all_embeddings = kelpie_model.entity_embeddings.cpu().numpy()
    kelpie_embedding = all_embeddings[-1]
    original_embedding = all_embeddings[original_entity_id]

    original_distances = []
    kelpie_distances = []
    for i in range(kelpie_dataset.num_entities):
        if i != original_entity_id and i != kelpie_entity_id :
            original_distances.append(numpy.linalg.norm(all_embeddings[i] - original_embedding, 2))
            kelpie_distances.append(numpy.linalg.norm(all_embeddings[i] - kelpie_embedding, 2))

    print("\tAverage distance of all entities from Barack Obama: %f (min: %f, max: %f)" %
          (numpy.average(original_distances), min(original_distances), max(original_distances)) )
    print("\tAverage distance of all entities from Fake Obama: %f (min: %f, max: %f)" %
          (numpy.average(kelpie_distances), min(kelpie_distances), max(kelpie_distances)) )
    print("\tDistance between original entity and kelpie entity:" + str(numpy.linalg.norm(original_embedding-kelpie_embedding, 2)))

#print("\nEnsuring that weights were actually frozen, and that the Kelpie entity weight was actually learned...")
#identical = kelpie_model.entities_weights.data[0:kelpie_dataset.n_entities] == original_model.entity_embeddings.data[0:kelpie_dataset.n_entities]

print("\nDone.")
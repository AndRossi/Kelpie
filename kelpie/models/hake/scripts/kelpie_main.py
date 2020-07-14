import argparse
import torch
import numpy

from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.kelpie_dataset import KelpieDataset
from kelpie.evaluation import Evaluator, KelpieEvaluator
from kelpie.models.hake.data import get_train_iterator_from_dataset
from kelpie.models.hake.model import Hake, KelpieHake
from kelpie.models.hake.optimizer import HakeOptimizer

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
                    default='Adam',
                    help="Optimizer in {} to use in post-training".format(optimizers))

parser.add_argument('--batch_size',
                    default=1024,
                    type=int,
                    help="Batch size to use in post-training")

parser.add_argument('--test_batch_size',
                    default=4,
                    type=int,
                    help="Number of samples in each mini-batch in SGD, Adagrad and Adam optimization during evaluation"
)

parser.add_argument('--max_epochs',
                    default=20,
                    type=int,
                    help="Number of epochs to run in post-training")

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Factorization rank.")

parser.add_argument('--learning_rate',
                    default=0.0001,
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

# Hake-specific args:

parser.add_argument('--gamma',
                    default=1.0,
                    type=float,
                    help="Gamma"
)

parser.add_argument('--modulus_weight',
                    default=1.0,
                    type=float,
                    help="Modulus weight"
)

parser.add_argument('--phase_weight',
                    default=0.5,
                    type=float,
                    help="Phase weight"
)

parser.add_argument('--no_decay',
                    default=False,
                    type=bool,
                    help="Disables decay if true"
)

parser.add_argument('--adversarial_temperature',
                    default=1.0,
                    type=float,
                    help="Adversarial temperature"
)

parser.add_argument('--cpu_num',
                    default=10,
                    type=int,
                    help="Number of (virtual) CPU cores to use"
)

parser.add_argument('--negative_sample_size',
                    default=128,
                    type=int,
                    help = "Number of negative samples"
)

parser.add_argument('--init_step',
                    default=0,
                    type=int,
                    help="Initial training step number"
)

parser.add_argument('--valid',
                    default=-1,
                    type=float,
                    help="Number of epochs before valid."
)

#

#   E.G.    explain  why  /m/02mjmr (Barack Obama)  is predicted as the head for
#   /m/02mjmr (Barack Obama) 	/people/person/ethnicity	/m/033tf_ (Irish American)
args = parser.parse_args()


#############  LOAD DATASET

# get the fact to explain and its perspective entity
head, relation, tail = args.head, args.relation, args.tail
entity_to_explain = head if args.perspective.lower() == "head" else tail

# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
original_dataset = Dataset(name=args.dataset,
                           separator="\t",
                           load=True)

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
original_model = Hake(dataset=original_dataset, hidden_dim=args.dimension, batch_size=args.batch_size, test_batch_size=args.test_batch_size,
             cpu_num=args.cpu_num, gamma=args.gamma, modulus_weight=args.modulus_weight, phase_weight=args.phase_weight)
original_model.load_state_dict(torch.load(args.model_path))
original_model.to('cuda')

print("Wrapping the original model in a Kelpie explainable model...")
# use model_to_explain to initialize the Kelpie model
kelpie_dataset = KelpieDataset(dataset=original_dataset, entity_id=original_entity_id)
kelpie_model = KelpieHake(dataset=kelpie_dataset, model=original_model)
kelpie_model.to('cuda')


############ EXTRACT TEST FACTS AND TRAINING FACTS

print("Extracting train and test samples for the original and the kelpie entities...")
# extract all training facts and test facts involving the entity to explain
# and replace the id of the entity to explain with the id of the fake kelpie entity
original_entity_test_samples = kelpie_dataset.original_test_samples
kelpie_test_samples = kelpie_dataset.kelpie_test_samples


###########  BUILD THE OPTIMIZER AND RUN POST-TRAINING

print("Running post-training on the Kelpie model...")
# build the Optimizer
optimizer = HakeOptimizer(model=kelpie_model,
                             optimizer_name=args.optimizer,
                             learning_rate=args.learning_rate,
                             no_decay=args.no_decay,
                             max_steps=args.max_epochs,
                             adversarial_temperature=args.adversarial_temperature)

train_iterator = get_train_iterator_from_dataset(triples=kelpie_dataset.kelpie_train_samples,
                                    num_entities=kelpie_dataset.num_entities,
                                    num_relations=kelpie_dataset.num_relations,
                                    cpu_num=args.cpu_num,
                                    batch_size=args.batch_size,
                                    negative_sample_size=args.negative_sample_size)

optimizer.train(train_iterator=train_iterator,
                init_step=args.init_step,
                evaluate_every=args.valid,
                valid_samples=kelpie_dataset.kelpie_valid_samples)

###########  EXTRACT RESULTS

print("\nExtracting results...")
kelpie_entity_id = kelpie_dataset.kelpie_entity_id
kelpie_sample_head = kelpie_entity_id if head_id == original_entity_id else head_id
kelpie_sample_tail = kelpie_entity_id if tail_id == original_entity_id else tail_id
kelpie_sample_tuple = (kelpie_sample_head, relation_id, kelpie_sample_tail)
kelpie_sample = numpy.array(kelpie_sample_tuple)

### Evaluation on original entity

# Original model on original fact
scores, ranks, _ = original_model.predict_sample(sample=original_sample)
print("\nOriginal model on original test fact: <%s, %s, %s>" % original_sample_tuple)
print("\tDirect fact score: %f; Inverse fact score: %f" % (scores[0], scores[1]))
print("\tHead Rank: %f" % ranks[0])
print("\tTail Rank: %f" % ranks[1])

# Original model on all facts containing the original entity
print("\nOriginal model on all test facts containing the original entity:")
mrr, h1 = Evaluator(original_model).eval(samples=original_entity_test_samples)
print("\tMRR: %f\n\tH@1: %f" % (mrr, h1))

# Kelpie model model on original fact
scores, ranks, _ = kelpie_model.predict_sample(sample=original_sample, original_mode=True)
print("\nKelpie model on original test fact: <%s, %s, %s>" % original_sample_tuple)
print("\tDirect fact score: %f; Inverse fact score: %f" % (scores[0], scores[1]))
print("\tHead Rank: %f" % ranks[0])
print("\tTail Rank: %f" % ranks[1])

# Kelpie model on all facts containing the original entity
print("\nKelpie model on all test facts containing the original entity:")
mrr, h1 = KelpieEvaluator(kelpie_model).eval(samples=original_entity_test_samples, original_mode=True)
print("\tMRR: %f\n\tH@1: %f" % (mrr, h1))


### Evaluation on kelpie entity

# results on kelpie fact
scores, ranks, _ = kelpie_model.predict_sample(sample=kelpie_sample, original_mode=False)
print("\nKelpie model on original test fact: <%s, %s, %s>" % kelpie_sample_tuple)
print("\tDirect fact score: %f; Inverse fact score: %f" % (scores[0], scores[1]))
print("\tHead Rank: %f" % ranks[0])
print("\tTail Rank: %f" % ranks[1])

# results on all facts containing the kelpie entity
print("\nKelpie model on all test facts containing the Kelpie entity:")
mrr, h1 = KelpieEvaluator(kelpie_model).eval(samples=kelpie_test_samples, original_mode=False)
print("\tMRR: %f\n\tH@1: %f" % (mrr, h1))

#print("\n\nComputing embedding distances...")
#with torch.no_grad():
#    all_embeddings = kelpie_model.entity_embeddings.cpu().numpy()
#    kelpie_embedding = all_embeddings[-1]
#    original_embedding = all_embeddings[original_entity_id]
#
#    original_distances = []
#    kelpie_distances = []
#    for i in range(kelpie_dataset.num_entities):
#        if i != original_entity_id and i != kelpie_entity_id :
#            original_distances.append(numpy.linalg.norm(all_embeddings[i] - original_embedding, 2))
#            kelpie_distances.append(numpy.linalg.norm(all_embeddings[i] - kelpie_embedding, 2))
#
#    print("\tAverage distance of all entities from Barack Obama: %f (min: %f, max: %f)" %
#          (numpy.average(original_distances), min(original_distances), max(original_distances)) )
#    print("\tAverage distance of all entities from Fake Obama: %f (min: %f, max: %f)" %
#          (numpy.average(kelpie_distances), min(kelpie_distances), max(kelpie_distances)) )
#    print("\tDistance between original entity and kelpie entity:" + str(numpy.linalg.norm(original_embedding-kelpie_embedding, 2)))

#print("\nEnsuring that weights were actually frozen, and that the Kelpie entity weight was actually learned...")
#identical = kelpie_model.entities_weights.data[0:kelpie_dataset.n_entities] == original_model.entity_embeddings.data[0:kelpie_dataset.n_entities]

print("\nDone.")
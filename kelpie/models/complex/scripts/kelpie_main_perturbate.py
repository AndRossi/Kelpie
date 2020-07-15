"""
This module does explanation for ComplEx model
"""
import argparse
import torch
import numpy

from kelpie import kelpie_perturbation
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import KelpieEvaluator
from kelpie.kelpie_dataset import KelpieDataset
from kelpie.models.complex.model import ComplEx, KelpieComplEx
from kelpie.models.complex.optimizer import KelpieComplExOptimizer

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
head, relation, tail = args.head, args.relation, args.tail
entity_to_explain = head if args.perspective.lower() == "head" else tail

# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
original_dataset = Dataset(name=args.dataset, separator="\t", load=True)

# get the ids of the elements of the fact to explain and the perspective entity
head_id, relation_id, tail_id = original_dataset.get_id_for_entity_name(head), \
                                original_dataset.get_id_for_relation_name(relation), \
                                original_dataset.get_id_for_entity_name(tail)
original_entity_id = head_id if args.perspective == "head" else tail_id

# create the fact to explain as a numpy array of its ids
original_triple = (head_id, relation_id, tail_id)
original_sample = numpy.array(original_triple)

# check that the fact to explain is actually a test fact
assert(original_sample in original_dataset.test_samples)


#############   INITIALIZE MODELS AND THEIR STRUCTURES
print("Loading model at location %s..." % args.model_path)
# instantiate and load the original model from filesystem
original_model = ComplEx(dataset=original_dataset,
                         dimension=args.dimension,
                         init_random=True,
                         init_size=args.init) # type: ComplEx
original_model.load_state_dict(torch.load(args.model_path))
original_model.to('cuda')

kelpie_dataset = KelpieDataset(dataset=original_dataset, entity_id=original_entity_id)


############ EXTRACT TEST FACTS AND TRAINING FACTS

print("Extracting train and test samples for the original and the kelpie entities...")
# extract all training facts and test facts involving the entity to explain
# and replace the id of the entity to explain with the id of the fake kelpie entity
original_test_samples = kelpie_dataset.original_test_samples
kelpie_test_samples = kelpie_dataset.kelpie_test_samples
kelpie_train_samples = kelpie_dataset.kelpie_train_samples

perturbed_list, skipped_list = kelpie_perturbation.perturbate_samples(kelpie_train_samples)

def run_kelpie(train_samples):
    print("Wrapping the original model in a Kelpie explainable model...")
    # use model_to_explain to initialize the Kelpie model
    kelpie_model = KelpieComplEx(model=original_model, dataset=kelpie_dataset, init_size=1e-3) # type: KelpieComplEx
    kelpie_model.to('cuda')

    ###########  BUILD THE OPTIMIZER AND RUN POST-TRAINING
    print("Running post-training on the Kelpie model...")
    optimizer = KelpieComplExOptimizer(model=kelpie_model,
                                       optimizer_name=args.optimizer,
                                       batch_size=args.batch_size,
                                       learning_rate=args.learning_rate,
                                       decay1=args.decay1,
                                       decay2=args.decay2,
                                       regularizer_name="N3",
                                       regularizer_weight=args.reg)
    optimizer.train(train_samples=train_samples, max_epochs=args.max_epochs)

    ###########  EXTRACT RESULTS

    print("\nExtracting results...")
    kelpie_entity_id = kelpie_dataset.kelpie_entity_id
    kelpie_sample_tuple = (kelpie_entity_id, relation_id, tail_id) if args.perspective == "head" else (head_id, relation_id, kelpie_entity_id)
    kelpie_sample = numpy.array(kelpie_sample_tuple)

    ### Evaluation on original entity

    # Kelpie model on original fact
    scores, ranks, predictions = kelpie_model.predict_sample(sample=original_sample, original_mode=True)
    original_direct_score, original_inverse_score = scores[0], scores[1]
    original_direct_rank, original_inverse_rank = ranks[0], [1]
    print("\nKelpie model on original test fact: <%s, %s, %s>" % original_triple)
    print("\tDirect fact score: %f; Inverse fact score: %f" % (original_direct_score, original_inverse_score))
    print("\tHead Rank: %f" % ranks[0])
    print("\tTail Rank: %f" % ranks[1])

    # Kelpie model on all facts containing the original entity
    print("\nKelpie model on all test facts containing the original entity:")
    mrr, h1 = KelpieEvaluator(kelpie_model).eval(samples=original_test_samples, original_mode=True)
    print("\tMRR: %f\n\tH@1: %f" % (mrr, h1))


    ### Evaluation on kelpie entity

    # results on kelpie fact
    scores, ranks, _ = kelpie_model.predict_sample(sample=kelpie_sample, original_mode=False)
    kelpie_direct_score, kelpie_inverse_score = scores[0], scores[1]
    kelpie_direct_rank, kelpie_inverse_rank = ranks[0], ranks[1]
    print("\nKelpie model on original test fact: <%s, %s, %s>" % kelpie_sample_tuple)
    print("\tDirect fact score: %f; Inverse fact score: %f" % (kelpie_direct_score, kelpie_inverse_score))
    print("\tHead Rank: %f" % ranks[0])
    print("\tTail Rank: %f" % ranks[1])

    # results on all facts containing the kelpie entity
    print("\nKelpie model on all test facts containing the Kelpie entity:")
    mrr, h1 = KelpieEvaluator(kelpie_model).eval(samples=kelpie_test_samples, original_mode=False)
    print("\tMRR: %f\n\tH@1: %f" % (mrr, h1))

    return kelpie_direct_score, kelpie_inverse_score, kelpie_direct_rank, kelpie_inverse_rank



outlines = []
print("### BASE POST-TRAINING")
direct_score, inverse_score, direct_rank, inverse_rank = run_kelpie(kelpie_train_samples)
outlines.append("\t".join(["NO FACTS SKIPPED",
                           str(direct_score), str(inverse_score), str(direct_rank), str(inverse_rank)]) + "\n")

for i in range(len(perturbed_list)):
    samples = perturbed_list[i]

    skipped_samples = skipped_list[i]
    skipped_facts = []
    for s in skipped_samples:
        fact = (kelpie_dataset.entity_id_2_name[int(s[0])],
                kelpie_dataset.relation_id_2_name[int(s[1])],
                kelpie_dataset.entity_id_2_name[int(s[2])])
        skipped_facts.append(";".join(fact))

    print("### ITER %i" % i)

    print("### SKIPPED FACTS: ")
    for x in skipped_facts:
        print("\t" + x)

    cur_direct_score, cur_inverse_score, cur_direct_rank, cur_inverse_rank = run_kelpie(samples)

    direct_diff = cur_direct_score - direct_score
    inverse_diff = cur_inverse_score - inverse_score

    outlines.append("\t".join([skipped_facts[0], str(cur_direct_score), str(cur_inverse_score), str(cur_direct_rank), str(cur_inverse_rank)]) + "\n")

with open("output.txt", "w") as outfile:
    outfile.writelines(outlines)
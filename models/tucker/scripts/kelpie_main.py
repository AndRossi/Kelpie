import argparse
import torch
import numpy

from dataset import ALL_DATASET_NAMES, Dataset, KelpieDataset
from evaluation import Evaluator, KelpieEvaluator
from models.tucker.model import TuckER, KelpieTuckER
from optimization.bce_optimizer import KelpieBCEOptimizer

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument('--dataset',
                    choices=datasets,
                    help="Dataset in {}".format(datasets),
                    required=True)

parser.add_argument('--model_path',
                    help="Path to the model to explain the predictions of",
                    required=True)

parser.add_argument('--batch_size',
                    default=128,
                    type=int,
                    help="Batch size to use in post-training")

parser.add_argument('--max_epochs',
                    default=500,
                    type=int,
                    help="Number of epochs to run in post-training")

parser.add_argument('--entity_dimension',
                    default=200,
                    type=int,
                    help="dimensionality of entity embeddings.")

parser.add_argument('--relation_dimension',
                    default=200,
                    type=int,
                    help="dimensionality of relation embeddings.")

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate")

parser.add_argument("--input_dropout",
                    default=0.3,
                    type=float,
                    help="Input layer dropout.")

parser.add_argument("--hidden_dropout1",
                    default=0.4,
                    type=float,
                    help="Dropout after the first hidden layer.")

parser.add_argument("--hidden_dropout2",
                    default = 0.3,
                    type = float,
                    help="Dropout after the second hidden layer.")

parser.add_argument("--label_smoothing",
                    default=0.1,
                    type=float,
                    help="Amount of label smoothing.")

parser.add_argument('--decay_rate',
                    default=1.0,
                    type=float,
                    help="Decay rate"
)

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
original_model = TuckER(dataset=original_dataset,
                        entity_dimension=args.entity_dimension, relation_dimension=args.relation_dimension,
                        input_dropout=args.input_dropout,
                        hidden_dropout1=args.hidden_dropout1,
                        hidden_dropout2=args.hidden_dropout2,
                        init_random=True)  # type: TuckER
original_model.load_state_dict(torch.load(args.model_path))
original_model.to('cuda')
original_model.eval()

print("Wrapping the original model in a Kelpie explainable model...")
# use model_to_explain to initialize the Kelpie model
kelpie_dataset = KelpieDataset(dataset=original_dataset, entity_id=original_entity_id)
kelpie_model = KelpieTuckER(model=original_model, dataset=kelpie_dataset,
                            entity_dimension=args.entity_dimension, relation_dimension=args.relation_dimension,
                            input_dropout=args.input_dropout,
                            hidden_dropout1=args.hidden_dropout1,
                            hidden_dropout2=args.hidden_dropout2) # type: KelpieTuckER
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
optimizer = KelpieBCEOptimizer(model=kelpie_model,
                               batch_size=args.batch_size,
                               learning_rate=args.learning_rate,
                               decay=args.decay_rate,
                               label_smoothing=args.label_smoothing)

optimizer.train(train_samples=kelpie_dataset.kelpie_train_samples,
                max_epochs=args.max_epochs)

###########  EXTRACT RESULTS
original_model.eval()
kelpie_model.eval()

print("\nExtracting results...")
kelpie_entity_id = kelpie_dataset.kelpie_entity_id
kelpie_sample_head = kelpie_entity_id if head_id == original_entity_id else head_id
kelpie_sample_tail = kelpie_entity_id if tail_id == original_entity_id else tail_id
kelpie_sample_tuple = (kelpie_sample_head, relation_id, kelpie_sample_tail)
kelpie_sample = numpy.array(kelpie_sample_tuple)

### Evaluation on original entity

assert (original_model.core_tensor.detach().data == kelpie_model.core_tensor.detach().data).all()
assert (original_model.entity_embeddings[0:14951] == kelpie_model.entity_embeddings[0:14951]).all()
assert (original_model.relation_embeddings.detach().data == kelpie_model.relation_embeddings.detach().data).all()

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

print("\nDone.")
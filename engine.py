import numpy
import torch

from dataset import KelpieDataset, Dataset
from link_prediction.evaluation.evaluation import KelpieEvaluator
from link_prediction.optimization.multiclass_nll_optimizer import KelpieMultiClassNLLOptimizer
from link_prediction.perturbation import kelpie_perturbation
from model import Model, KelpieModel

class PostTrainingEngine:

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        """
            PostTrainingEngine constructor.

            :param model: the trained Model to explain the behaviour of. This can NOT be a KelpieModel.
            :param dataset: the Dataset used to train the model
            :param hyperparameters: dict containing all the hyperparameters necessary for running the post-training
                                    (for both the model and the optimizer)
        """

        if isinstance(model, KelpieModel):
            raise Exception("The model passed to the PostTrainingEngine is already a post-trainable KelpieModel.")

        self.model = model
        self.model.to('cuda')   # it this hasn't been done yet, load the model in GPU
        self.dataset = dataset
        self.hyperparameters = hyperparameters

    # TODO: somehow make perturbation policy parametric
    def run(self,
            sample_to_explain: numpy.array,
            perspective: str):

        # the behaviour of the engine must be deterministic!
        seed = 42
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True

        # sample to explain as a numpy array of the ids
        head_id, relation_id, tail_id = sample_to_explain
        # name and id of the entity to explain, based on the passed perspective
        original_entity_id = head_id if perspective == "head" else tail_id

        # create a Kelpie Dataset focused on the original id of the entity to explain
        kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_id=original_entity_id)

        print("### BASE POST-TRAINING")
        post_trained_model = self.post_train(kelpie_dataset=kelpie_dataset,
                                             kelpie_train_samples=kelpie_dataset.kelpie_train_samples)

        direct_score, inverse_score, head_rank, tail_rank = self.test_kelpie_model(testing_model=post_trained_model,
                                                                                   sample_to_explain=sample_to_explain,
                                                                                   perspective=perspective)

        print("Perturbing entities...")
        perturbed_list, skipped_list = kelpie_perturbation.perturbate_samples(kelpie_dataset.kelpie_train_samples)

        results = []
        for i in range(len(perturbed_list)):
            cur_training_samples = perturbed_list[i]

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

            cur_kelpie_model = self.post_train(kelpie_dataset=kelpie_dataset, kelpie_train_samples=cur_training_samples)
            cur_direct_score, cur_inverse_score, cur_head_rank, cur_tail_rank = self.test_kelpie_model(testing_model=cur_kelpie_model,
                                                                                                       sample_to_explain=sample_to_explain,
                                                                                                       perspective=perspective)
            direct_diff = direct_score - cur_direct_score
            inverse_diff = inverse_score - cur_inverse_score
            results.append((skipped_facts[0], cur_direct_score, cur_inverse_score, cur_head_rank, cur_tail_rank, direct_diff + inverse_diff))

        results = sorted(results, key=lambda x: x[-1], reverse=True)
        print("\t".join(["NO FACTS SKIPPED", str(direct_score), str(inverse_score), str(head_rank), str(tail_rank)]))
        print(results[:10])


    # todo: rename in a better way. This does not return the post-trained model, but just its results on the sample to explain
    def post_train(self,
                   kelpie_dataset: KelpieDataset,
                   kelpie_train_samples: numpy.array):
        """

        :param kelpie_dataset:
        :param kelpie_train_samples:
        :return:
        """
        print("Post-training the Kelpie model...")
        kelpie_model_clazz = self.model.kelpie_model_class()
        kelpie_model = kelpie_model_clazz(model=self.model, dataset=kelpie_dataset)
        kelpie_model.to('cuda')

        # todo: use the optimizer that works with the specific model.
        # todo: Maybe have it parametric? or maybe just have an official model-optimizer map in model.py?
        optimizer = KelpieMultiClassNLLOptimizer(model=kelpie_model, hyperparameters=self.hyperparameters)
        optimizer.train(train_samples=kelpie_train_samples)
        return kelpie_model

    def test_kelpie_model(self,
                          testing_model: KelpieModel,
                          sample_to_explain: numpy.array,
                          perspective: str):
        """

        :param testing_model:
        :param sample_to_explain:
        :param perspective:
        :return:
        """
        testing_model.eval()
        kelpie_dataset = testing_model.dataset   # type: KelpieDataset

        # Kelpie model on original fact
        scores, ranks, predictions = testing_model.predict_sample(sample=sample_to_explain, original_mode=True)
        original_direct_score, original_inverse_score = scores[0], scores[1]
        original_head_rank, original_tail_rank = ranks[0], ranks[1]
        print("\nKelpie model on the original fact <%s, %s, %s>" % (kelpie_dataset.entity_id_2_name[sample_to_explain[0]],
                                                                    kelpie_dataset.relation_id_2_name[sample_to_explain[1]],
                                                                    kelpie_dataset.entity_id_2_name[sample_to_explain[2]]))
        print("\tDirect fact score: %f; Inverse fact score: %f" % (original_direct_score, original_inverse_score))
        print("\tHead Rank: %f" % original_head_rank)
        print("\tTail Rank: %f" % original_tail_rank)

        # Kelpie model on all facts containing the original entity
        print("\nKelpie model on all test facts containing the original entity:")
        mrr, h1 = KelpieEvaluator(testing_model).eval(samples=kelpie_dataset.original_test_samples, original_mode=True)
        print("\tMRR: %f\n\tH@1: %f" % (mrr, h1))

        ### Evaluation on kelpie entity
        kelpie_entity_id = kelpie_dataset.kelpie_entity_id
        kelpie_sample = numpy.array((kelpie_entity_id, sample_to_explain[1], sample_to_explain[2])) if perspective == "head" \
            else numpy.array((sample_to_explain[0], sample_to_explain[1], kelpie_entity_id))

        # results on kelpie fact
        scores, ranks, predictions = testing_model.predict_sample(sample=kelpie_sample, original_mode=False)
        kelpie_direct_score, kelpie_inverse_score = scores[0], scores[1]
        kelpie_head_rank, kelpie_tail_rank = ranks[0], ranks[1]
        print("\nKelpie model on the Kelpie test fact: <%s, %s, %s>" % (kelpie_dataset.entity_id_2_name[kelpie_sample[0]],
                                                                        kelpie_dataset.relation_id_2_name[kelpie_sample[1]],
                                                                        kelpie_dataset.entity_id_2_name[kelpie_sample[2]]))
        print("\tDirect fact score: %f; Inverse fact score: %f" % (kelpie_direct_score, kelpie_inverse_score))
        print("\tHead Rank: %f" % kelpie_head_rank)
        print("\tTail Rank: %f" % kelpie_tail_rank)

        # results on all facts containing the kelpie entity
        print("\nKelpie model on all test facts containing the Kelpie entity:")
        mrr, h1 = KelpieEvaluator(testing_model).eval(samples=kelpie_dataset.kelpie_test_samples, original_mode=False)
        print("\tMRR: %f\n\tH@1: %f" % (mrr, h1))

        return kelpie_direct_score, kelpie_inverse_score, kelpie_head_rank, kelpie_tail_rank

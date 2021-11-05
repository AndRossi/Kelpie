import tqdm
import torch
import numpy as np
from torch import optim
from link_prediction.regularization.regularizers import L2
from link_prediction.models.model import Model, BATCH_SIZE, LEARNING_RATE, EPOCHS, MARGIN, NEGATIVE_SAMPLES_RATIO, \
    REGULARIZER_WEIGHT, KelpieModel
from link_prediction.optimization.optimizer import Optimizer


class PairwiseRankingOptimizer(Optimizer):
    """
        This optimizer relies on Pairwise Ranking loss, also called Margin-based loss.
    """

    def __init__(self,
                 model: Model,
                 hyperparameters: dict,
                 verbose: bool = True):
        """
            PairwiseRankingOptimizer initializer.
            :param model: the model to train
            :param hyperparameters: a dict with the optimization hyperparameters. It must contain at least:
                    - BATCH SIZE
                    - LEARNING RATE
                    - DECAY
                    - LABEL SMOOTHING
                    - EPOCHS
            :param verbose:
        """

        Optimizer.__init__(self, model=model, hyperparameters=hyperparameters, verbose=verbose)

        self.batch_size = hyperparameters[BATCH_SIZE]
        self.learning_rate = hyperparameters[LEARNING_RATE]
        self.epochs = hyperparameters[EPOCHS]
        self.margin = hyperparameters[MARGIN]
        self.negative_samples_ratio = hyperparameters[NEGATIVE_SAMPLES_RATIO]
        self.regularizer_weight = hyperparameters[REGULARIZER_WEIGHT]

        #self.corruption_engine = CorruptionEngine(self.dataset)
        self.loss = torch.nn.MarginRankingLoss(margin=self.margin, reduction="mean").cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)  # we only support ADAM for PairwiseRankingOptimizer
        self.regularizer = L2(self.regularizer_weight)

    def train(self,
              train_samples: np.array,
              save_path: str = None,
              evaluate_every:int =-1,
              valid_samples:np.array = None):

        training_samples = np.vstack((train_samples, self.dataset.invert_samples(train_samples)))

        self.model.cuda()

        for e in range(1, self.epochs+1):

            self.epoch(training_samples=training_samples, batch_size=self.batch_size)

            if evaluate_every > 0 and valid_samples is not None and e % evaluate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    mrr, h1, h10, mr = self.evaluator.evaluate(samples=valid_samples, write_output=False)

                print("\tValidation Hits@1: %f" % h1)
                print("\tValidation Hits@10: %f" % h10)
                print("\tValidation Mean Reciprocal Rank': %f" % mrr)
                print("\tValidation Mean Rank': %f" % mr)

                if save_path is not None:
                    print("\t saving model...")
                    torch.save(self.model.state_dict(), save_path)
                print("\t done.")

        if save_path is not None:
            print("\t saving model...")
            torch.save(self.model.state_dict(), save_path)
            print("\t done.")


    def epoch(self,
              batch_size: int,
              training_samples: np.array):

        np.random.shuffle(training_samples)
        repeated = np.repeat(training_samples, self.negative_samples_ratio, axis=0)

        all_positive_samples = torch.from_numpy(repeated)

        head_or_tail = torch.randint(high=2, size=torch.Size([len(all_positive_samples)]))
        random_entities = torch.randint(high=self.dataset.num_entities, size=torch.Size([len(all_positive_samples)]))

        corrupted_heads = torch.where(head_or_tail == 1, random_entities, all_positive_samples[:, 0].type(torch.LongTensor))
        corrupted_tails = torch.where(head_or_tail == 0, random_entities, all_positive_samples[:, 2].type(torch.LongTensor))
        all_negative_samples = torch.stack((corrupted_heads, all_positive_samples[:, 1], corrupted_tails), dim=1)

        all_positive_samples.cuda()
        all_negative_samples.cuda()
        self.model.train()

        with tqdm.tqdm(total=len(training_samples), unit='ex', disable=not self.verbose) as bar:
            bar.set_description('train loss')
            batch_start = 0

            while batch_start < len(training_samples):
                positive_batch, negative_batch = all_positive_samples[batch_start: min(batch_start+batch_size, len(training_samples))], \
                                                 all_negative_samples[batch_start: min(batch_start+batch_size, len(training_samples))]

                l = self.step_on_batch(positive_batch, negative_batch)
                batch_start+=batch_size
                bar.update(batch_size)
                bar.set_postfix(loss=str(round(l.item(), 6)))

    def step_on_batch(self, positive_batch, negative_batch):

        self.optimizer.zero_grad()

        positive_scores, positive_factors = self.model.forward(positive_batch)
        negative_scores, negative_factors = self.model.forward(negative_batch)
        target = torch.tensor([-1], dtype=torch.float).cuda()

        l_fit = self.loss(positive_scores, negative_scores, target)
        l_reg = (self.regularizer.forward(positive_factors) + self.regularizer.forward(negative_factors))/2

        loss = l_fit + l_reg
        loss.backward()
        self.optimizer.step()

        return loss

class KelpiePairwiseRankingOptimizer(PairwiseRankingOptimizer):
    def __init__(self,
                 model:KelpieModel,
                 hyperparameters: dict,
                 verbose: bool = True):

        super(KelpiePairwiseRankingOptimizer, self).__init__(model=model,
                                                             hyperparameters=hyperparameters,
                                                             verbose=verbose)
        self.kelpie_entity_id = model.kelpie_entity_id

    # Override
    def epoch(self,
              batch_size: int,
              training_samples: np.array):

        np.random.shuffle(training_samples)
        self.model.train()

        repeated = np.repeat(training_samples, self.negative_samples_ratio, axis=0)
        all_positive_samples = torch.from_numpy(repeated)

        random_entities = torch.randint(high=self.dataset.num_entities, size=torch.Size([len(all_positive_samples)]))
        head_or_tail = torch.randint(high=2, size=torch.Size([len(all_positive_samples)]))
        corrupted_heads = torch.where(head_or_tail == 1, random_entities, all_positive_samples[:, 0].type(torch.LongTensor))
        corrupted_tails = torch.where(head_or_tail == 0, random_entities, all_positive_samples[:, 2].type(torch.LongTensor))

        # kelpie_entity_is_head = torch.where(all_positive_samples[:, 0] == self.kelpie_entity_id, torch.ones(len(all_positive_samples)), torch.zeros(len(all_positive_samples)))
        # corrupted_heads = torch.where(kelpie_entity_is_head == 0, random_entities, all_positive_samples[:, 0].type(torch.LongTensor))
        # corrupted_tails = torch.where(kelpie_entity_is_head == 1, random_entities, all_positive_samples[:, 2].type(torch.LongTensor))

        all_negative_samples = torch.stack((corrupted_heads, all_positive_samples[:, 1], corrupted_tails), dim=1)

        all_positive_samples.cuda()
        all_negative_samples.cuda()

        with tqdm.tqdm(total=len(training_samples), unit='ex', disable=not self.verbose) as bar:
            bar.set_description('train loss')

            batch_start = 0

            while batch_start < len(training_samples):
                positive_batch, negative_batch = all_positive_samples[batch_start: min(batch_start+batch_size, len(training_samples))], \
                                                 all_negative_samples[batch_start: min(batch_start+batch_size, len(training_samples))]

                l = self.step_on_batch(positive_batch, negative_batch)

                # THIS IS THE ONE DIFFERENCE FROM THE ORIGINAL OPTIMIZER.
                # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES IN THE KELPIE ENTITY EMBEDDING
                # TO THE MATRIX CONTAINING ALL THE EMBEDDINGS
                self.model.update_embeddings()

                batch_start+=batch_size
                bar.update(batch_size)
                bar.set_postfix(loss=str(round(l.item(), 6)))
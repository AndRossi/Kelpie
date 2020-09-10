from torch.nn.modules.loss import _Loss
import tqdm
import torch
import numpy as np
from torch import optim

from model import Model
from evaluation import Evaluator


class PairwiseRankingOptimizer:
    """
        This optimizer relies on Pairwise Ranking loss.
    """

    def __init__(self,
                 model: Model,
                 margin: float,
                 optimizer_name: str = "Adagrad",
                 batch_size :int = 256,
                 learning_rate: float = 1e-2,
                 decay1 :float = 0.9,
                 decay2 :float = 0.99,
                 verbose: bool = True):
        self.model = model
        self.batch_size = batch_size
        self.verbose = verbose
        self.margin = margin
        # build all the supported optimizers using the passed params (learning rate and decays if Adam)

        supported_optimizers = {
            'Adagrad': optim.Adagrad(params=self.model.parameters(), lr=learning_rate),
            'Adam': optim.Adam(params=self.model.parameters(), lr=learning_rate, betas=(decay1, decay2)),
            'SGD': optim.SGD(params=self.model.parameters(), lr=learning_rate)
        }


        # choose the Torch Optimizer object to use, based on the passed name
        self.optimizer = supported_optimizers[optimizer_name]

        # create the evaluator to use between epochs
        self.evaluator = Evaluator(self.model)

    def train(self,
              train_samples: np.array,
              max_epochs: int,
              save_path: str = None,
              evaluate_every:int =-1,
              valid_samples:np.array = None):

        # extract the direct and inverse train facts
        training_samples = np.vstack((train_samples,
                                      self.model.dataset.invert_samples(train_samples)))

        # batch size must be the minimum between the passed value and the number of Kelpie training facts
        batch_size = min(self.batch_size, len(training_samples))

        cur_loss = 0
        for e in range(max_epochs):
            cur_loss = self.epoch(batch_size, training_samples)

            if evaluate_every > 0 and valid_samples is not None and \
                    (e + 1) % evaluate_every == 0:
                mrr, h1 = self.evaluator.eval(samples=valid_samples, write_output=False)

                print("\tValidation Hits@1: %f" % h1)
                print("\tValidation Mean Reciprocal Rank': %f" % mrr)

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
        training_samples = torch.from_numpy(training_samples).cuda()
        negative_samples = []

        # at the beginning of the epoch, shuffle all samples randomly
        permutation = torch.randperm(training_samples.shape[0])
        permuted_positive_samples = training_samples[permutation, :]
        permuted_negative_samples = negative_samples[permutation, :]

        loss = PairwiseRankingLoss(margin=self.margin)

        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch_end = min(batch_start + batch_size, training_samples.shape[0])

                positive_batch = permuted_positive_samples[batch_start : batch_end].cuda()
                negative_batch = permuted_negative_samples[batch_start : batch_end].cuda()
                l = self.step_on_batch(loss, positive_batch, negative_batch)

                batch_start += self.batch_size
                bar.update(positive_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')


    def step_on_batch(self, loss, positive_batch, negative_batch):
        positive_samples_scores = self.model.forward(positive_batch)
        negative_samples_scores = self.model.forward(negative_batch)

        # compute loss
        l_fit = loss(positive_samples_scores, negative_samples_scores)
        #l_reg = self.regularizer.forward(factors)
        l = l_fit #+ l_reg

        # compute loss gradients, and run optimization step
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        # return loss
        return l



class PairwiseRankingLoss(_Loss):
    def __init__(self, margin):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, positive_samples_scores: torch.Tensor,
                negative_samples_scores: torch.Tensor):
        return torch.sum(torch.max(self.margin + positive_samples_scores - negative_samples_scores), 0)


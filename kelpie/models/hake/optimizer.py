import tqdm
import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F

from kelpie.evaluation import Evaluator
from kelpie.models.hake.data import get_test_step_samples
from kelpie.models.hake.model import Hake


class HakeOptimizer:

    def __init__(self,
                 model: Hake,
                 optimizer_name: str = "Adam",
                 batch_size: int = 256,
                 learning_rate: float = 1e-2,
                 decay1: float = 0.9,
                 decay2: float = 0.99,
                 verbose: bool = True):
        self.model = model
        self.batch_size = batch_size
        self.verbose = verbose

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


    def _train(self,
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


    def train_step(self, model: torch.nn.modules.module, triple, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        self.optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = get_test_step_samples(triple)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
                        # gamma - dr(h, t)
        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

                        # p(h,r,t)                      # alpha
        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                            # E log sigma (dr(h, t) - gamma)
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        self.optimizer.step()

    def epoch(self,
              batch_size: int,
              training_samples: np.array):
        training_samples = torch.from_numpy(training_samples).cuda()

        # at the beginning of the epoch, shuffle all samples randomly
        actual_samples = training_samples[torch.randperm(training_samples.shape[0]), :]
        #loss = nn.CrossEntropyLoss(reduction='mean')
        loss = self._loss()

        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch = actual_samples[batch_start : batch_start + batch_size].cuda()
                l = self.step_on_batch(loss, batch)

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')


    def step_on_batch(self, loss, batch):
        # predictions, factors = self.model.forward(batch)
        predictions = self.model.forward(batch)
        truth = batch[:, 2]

        # compute loss
        # l_fit = loss(predictions, truth)
        # l_reg = self.regularizer.forward(factors)
        # l = l_fit + l_reg
        l = loss(predictions, truth)

        # compute loss gradients, and run optimization step
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        # return loss
        return l
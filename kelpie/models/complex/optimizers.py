import tqdm
import torch
import numpy as np
from torch import nn
from torch import optim

from kelpie.models.complex.complex import ComplEx, KelpieComplEx
from kelpie.models.complex.regularizers import Regularizer


class ComplExOptimizer:
    def __init__(self,
                 model: ComplEx,
                 regularizer: Regularizer,
                 optimizer: optim.Optimizer,
                 batch_size: int = 256,
                 verbose: bool = True):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, training_samples: np.array):
        training_samples = torch.from_numpy(training_samples).cuda()

        # at the beginning of the epoch, shuffle all samples randomly
        actual_samples = training_samples[torch.randperm(training_samples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')


        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch = actual_samples[batch_start : batch_start+self.batch_size].cuda()
                l = self.step_on_batch(loss, batch)

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')

    def step_on_batch(self, loss, batch):
        predictions, factors = self.model.forward(batch)
        truth = batch[:, 2]

        # compute loss
        l_fit = loss(predictions, truth)
        l_reg = self.regularizer.forward(factors)
        l = l_fit + l_reg

        # compute loss gradients, and run optimization step
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        # return loss
        return l


class KelpieComplExOptimizer(ComplExOptimizer):
    def __init__(self,
                 model: KelpieComplEx,
                 regularizer: Regularizer,
                 optimizer: optim.Optimizer,
                 batch_size: int = 256,
                 verbose: bool = True):

        super(KelpieComplExOptimizer, self).__init__(model=model,
                                                     regularizer=regularizer,
                                                     optimizer=optimizer,
                                                     batch_size=batch_size,
                                                     verbose=verbose)

    def epoch(self, training_samples: np.array):
        training_samples = torch.from_numpy(training_samples).cuda()
        # at the beginning of the epoch, shuffle all samples randomly
        actual_samples = training_samples[torch.randperm(training_samples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')

        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch = actual_samples[batch_start: batch_start + self.batch_size].cuda()
                l = self.step_on_batch(loss, batch)

                # THIS IS THE ONE DIFFERENCE FROM THE ORIGINAL OPTIMIZER.
                # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES IN THE KELPIE ENTITY EMBEDDING
                # TO THE MATRIX CONTAINING ALL THE EMBEDDINGS
                self.model.update_embeddings()

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
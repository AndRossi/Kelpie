# from helper import *
from ordered_set import OrderedSet
# from data_loader import *
# from model import *
import numpy as np
from kelpie.evaluation import Evaluator
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import tqdm

from kelpie.models.interacte.model import InteractE


class InteractEOptimizer:

    def __init__(self,
        model: InteractE,
        optimizer_name: str ="Adam",
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        decay_adam_1: float = 0.9,
        decay_adam_2: float = 0.99,
        weight_decay: float = 0.0, # Decay for Adam optimizer
        verbose: bool = True):

        self.model = model
        self.batch_size = batch_size
        self.verbose = verbose

        # Optimizer selection
        # build all the supported optimizers using the passed params (learning rate and decays if Adam)
        supported_optimizers = {
            'Adam': optim.Adam(params=self.model.parameters(), lr=learning_rate, betas=(decay_adam_1, decay_adam_2), weight_decay=weight_decay),
            'SGD': optim.SGD(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        }
        # choose which Torch Optimizer object to use, based on the passed name
        self.optimizer = supported_optimizers[optimizer_name]

        # create the evaluator to use between epochs
        self.evaluator = Evaluator(self.model)


    def train(
        self,
        train_samples: np.array, # fact triples
        max_epochs: int,
        save_path: str = None,
        evaluate_every: int = -1,
        valid_samples: np.array = None,
        strategy: str = 'one-to-n'):

        batch_size = min(self.batch_size, len(train_samples))
        
        cur_loss = 0
        
        for e in range(max_epochs):
            cur_loss = self.epoch(batch_size, train_samples)

            if evaluate_every > 0 and valid_samples is not None and (e + 1) % evaluate_every == 0:
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
        # at the beginning of the epoch, shuffle all samples randomly
        actual_samples = training_samples[torch.randperm(training_samples.shape[0]), :]
        loss = torch.nn.BCELoss()

        # Training over batches
        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch = actual_samples[batch_start : batch_start + batch_size].cuda()
                l = self.step_on_batch(loss, batch)

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')


    # Computing the loss over a single batch
    def step_on_batch(self, loss, batch):
        prediction = self.model.forward(batch)
        truth = batch[:, 2]
        oneHot_truth = F.one_hot(truth, prediction.shape[1]).type(torch.FloatTensor).cuda()
        
        # print('pred shape =', prediction.shape, '\t', type(prediction[0][0].item()))
        # print('truth shape =', truth.shape, '\t', type(truth[0].item()))
        # print('oneHot_truth shape =', oneHot_truth.shape, '\t', type(oneHot_truth[0][0].item()))
        
        # compute loss
        l = loss(prediction, oneHot_truth)
        
        # compute loss gradients and run optimization step
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        # return loss
        return l


class KelpieInteractEOptimizer(InteractEOptimizer):

    def __init__(self,
        model: InteractE,
        optimizer_name: str ="Adam",
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        decay_adam_1: float = 0.9,
        decay_adam_2: float = 0.99,
        weight_decay: float = 0.0, # Decay for Adam optimizer
        verbose: bool = True):
        
        super(KelpieInteractEOptimizer, self).__init__(
            model = model,
            optimizer_name = optimizer_name,
            batch_size = batch_size,
            learning_rate = learning_rate,
            decay_adam_1 = decay_adam_1,
            decay_adam_2 = decay_adam_2,
            weight_decay = weight_decay,
            verbose = verbose)

    
    def epoch(self,
            batch_size: int,
            training_samples: np.array):
        training_samples = torch.from_numpy(training_samples).cuda()
        # at the beginning of the epoch, shuffle all samples randomly
        actual_samples = training_samples[torch.randperm(training_samples.shape[0]), :]
        loss = torch.nn.BCELoss()

        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch = actual_samples[batch_start: batch_start + batch_size].cuda()
                l = self.step_on_batch(loss, batch)

                # THIS IS THE ONE DIFFERENCE FROM THE ORIGINAL OPTIMIZER.
                # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES IN THE KELPIE ENTITY EMBEDDING
                # TO THE MATRIX CONTAINING ALL THE EMBEDDINGS
                self.model.update_embeddings()

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
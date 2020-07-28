import tqdm
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from kelpie.models.interacte.model import InteractE, KelpieInteractE
from kelpie.evaluation import Evaluator


class InteractEOptimizer:

    def __init__(self,
                 model: InteractE,
                 optimizer_name: str = "Adam",
                 batch_size: int = 128,
                 learning_rate: float = 1e-4,
                 decay_adam_1: float = 0.9,
                 decay_adam_2: float = 0.99,
                 weight_decay: float = 0.0, # Decay for Adam optimizer
                 label_smooth: float = 0.1,
                 verbose: bool = True):

        self.model = model
        self.batch_size = batch_size
        self.label_smooth = label_smooth
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


    def train(self,
              train_samples: np.array, # fact triples
              max_epochs: int,
              save_path: str = None,
              evaluate_every: int = -1,
              valid_samples: np.array = None):

        # extract the direct and inverse train facts
        training_samples = np.vstack( (train_samples, self.model.dataset.invert_samples(train_samples)) )

        batch_size = min(self.batch_size, len(training_samples))
        
        for e in range(max_epochs):
            self.model.train()
            self.epoch(batch_size, training_samples)

            if evaluate_every > 0 and valid_samples is not None and (e + 1) % evaluate_every == 0:
                self.model.eval()
                with torch.no_grad():
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

            # losses = []
            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch_end = min(batch_start + batch_size, training_samples.shape[0])
                batch = actual_samples[batch_start : batch_end].cuda()
                
                l = self.step_on_batch(loss, batch)
                # losses.append(l)

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.5f}')
                
            # l = np.mean(losses)
            # bar.set_postfix(loss=f'{l.item():.5f}')


    # Computing the loss over a single batch
    def step_on_batch(self, loss, batch):
        prediction = self.model.forward(batch)
        truth = batch[:, 2]
        oneHot_truth = F.one_hot(truth, prediction.shape[1]).type(torch.FloatTensor).cuda()
        # multiHot_truth = torch.FloatTensor(batch.shape[0], prediction.shape[1]).cuda()
        # multiHot_truth = multiHot_truth.zero_().scatter_(1, truth, 1)
        
        # label smoothing
        oneHot_truth = (1.0 - self.label_smooth)*oneHot_truth + (1.0 / oneHot_truth.shape[1])
        
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
                 model: KelpieInteractE,
                 optimizer_name: str = "Adam",
                 batch_size: int = 128,
                 learning_rate: float = 1e-4,
                 decay_adam_1: float = 0.9,
                 decay_adam_2: float = 0.99,
                 weight_decay: float = 0.0, # Decay for Adam optimizer
                 label_smooth: float = 0.1,
                 verbose: bool = True):
        
        super(KelpieInteractEOptimizer, self).__init__(model = model,
                                                       optimizer_name = optimizer_name,
                                                       batch_size = batch_size,
                                                       learning_rate = learning_rate,
                                                       decay_adam_1 = decay_adam_1,
                                                       decay_adam_2 = decay_adam_2,
                                                       weight_decay = weight_decay,
                                                       label_smooth = label_smooth,
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
                batch_end = min(batch_start + batch_size, training_samples.shape[0])
                batch = actual_samples[batch_start : batch_end].cuda()
                
                l = self.step_on_batch(loss, batch)

                # THIS IS THE ONE DIFFERENCE FROM THE ORIGINAL OPTIMIZER.
                # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES IN THE KELPIE ENTITY EMBEDDING
                # TO THE MATRIX CONTAINING ALL THE EMBEDDINGS
                self.model.update_embeddings()

                batch_start += batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.5f}')
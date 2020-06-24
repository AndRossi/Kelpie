import tqdm
import torch
import numpy as np
from torch import nn
from torch import optim
from collections import defaultdict

from kelpie.models.tucker.model import TuckER, KelpieTuckER
from kelpie.evaluation import Evaluator

class TuckEROptimizer:
    def __init__(self,
                 model: TuckER,
                 optimizer_name: str = "Adam",
                 scheduler_name: str = "ExponentialLR",
                 batch_size: int = 128,
                 learning_rate: float = 0.0005,
                 decay: float = 1.0,
                 label_smoothing: float = 0.1,
                 verbose: bool = True):
        self.model = model
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.verbose = verbose

        # build all the supported optimizers using the passed params (learning rate if Adam)
        supported_optimizers = {
            'Adam': optim.Adam(params=self.model.parameters(), lr=learning_rate)
        }

        # choose the Torch Optimizer object to use, based on the passed name
        self.optimizer = supported_optimizers[optimizer_name]
        
        # build all the supported schedulers using the passed params (decay if ExponentialLR)
        supported_schedulers = {
            'ExponentialLR': optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        }
        
        # choose the Torch Scheduler object to use, based on the passed name
        self.scheduler = supported_schedulers[scheduler_name]

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
        
        er_vocab = self._get_er_vocab(training_samples)
        
        training_samples = torch.from_numpy(training_samples).cuda()

        # at the beginning of the epoch, shuffle all samples randomly
        actual_samples = training_samples[torch.randperm(training_samples.shape[0]), :]
        loss = nn.BCELoss()

        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch = actual_samples[batch_start : batch_start + batch_size].cuda()
                l = self.step_on_batch(loss, batch, er_vocab)

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')


    def step_on_batch(self, loss, batch, er_vocab):
        
        predictions = self.model.forward(batch)
        
        truth = self._get_truth(batch, er_vocab)
        truth = ((1.0-self.label_smoothing)*truth) + (1.0/truth.size(1))

        # compute loss
        l = loss(predictions, truth)

        # compute loss gradients, and run optimization step
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
        self.scheduler.step()

        # return loss
        return l

    # Build a dictionary which maps (head, relation) -> [tails]
    def _get_er_vocab(self, data):
        
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
            
        return er_vocab
    
    # Get a one-hot vector for each (head, relation) pair in batch.
    # The vector has 1.0 in indexes corresponding to the tails, and 0.0
    # elsewhere.
    def _get_truth(self, batch, er_vocab):
        
        truth = np.zeros((batch.shape[0], self.model.dataset.num_entities))
        for i, sample in enumerate(batch):
            head_relation_pair = (sample[0], sample[1])
            tails = er_vocab[head_relation_pair]
            truth[i, tails] = 1.
            
        truth = torch.FloatTensor(truth).cuda()
        
        return truth

class KelpieTuckEROptimizer(TuckEROptimizer):
    def __init__(self,
                 model:KelpieTuckER,
                 optimizer_name: str = "Adam",
                 scheduler_name: str = "ExponentialLR",
                 batch_size: int = 128,
                 learning_rate: float = 0.0005,
                 decay: float = 1.0,
                 label_smoothing: float = 0.1,
                 verbose: bool = True):

        super(KelpieTuckEROptimizer, self).__init__(model=model,
                                                    optimizer_name=optimizer_name,
                                                    scheduler_name=scheduler_name,
                                                    batch_size=batch_size,
                                                    learning_rate=learning_rate,
                                                    decay=decay,
                                                    label_smoothing=label_smoothing,
                                                    verbose=verbose)
# Override
    def epoch(self,
              batch_size: int,
              training_samples: np.array):
        training_samples = torch.from_numpy(training_samples).cuda()
        # at the beginning of the epoch, shuffle all samples randomly
        actual_samples = training_samples[torch.randperm(training_samples.shape[0]), :]
        loss = torch.nn.BCELoss()

        er_vocab = self._get_er_vocab(training_samples)

        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch = actual_samples[batch_start: batch_start + batch_size].cuda()
                l = self.step_on_batch(loss, batch, er_vocab)

                # THIS IS THE ONE DIFFERENCE FROM THE ORIGINAL OPTIMIZER.
                # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES IN THE KELPIE ENTITY EMBEDDING
                # TO THE MATRIX CONTAINING ALL THE EMBEDDINGS
                self.model.update_embeddings()

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
import tqdm
import torch
import numpy as np
from torch import optim
from collections import defaultdict
from link_prediction.evaluation.evaluation import Evaluator
from model import Model, BATCH_SIZE, LABEL_SMOOTHING, LEARNING_RATE, DECAY, EPOCHS


class BCEOptimizer:
    """
        This optimizer relies on BCE loss.
        Instead of considering each training sample as the "unit" for training,
        it groups training samples into couples (h, r) -> [all t for which <h, r, t> in training set].
        Each couple (h, r) with the corresponding tails is treated as if it was one sample.

        When passing them to the loss...

        In our implementation, it is used by the following models:
            - TuckER

    """

    def __init__(self,
                 model: Model,
                 hyperparameters: dict,
                 verbose: bool = True):

        """
            BCEOptimizer initializer.
            :param model: the model to train
            :param hyperparameters: a dict with the optimization hyperparameters. It must contain at least:
                    - BATCH SIZE
                    - LEARNING RATE
                    - DECAY
                    - LABEL SMOOTHING
                    - EPOCHS
            :param verbose:
        """

        #batch_size: int = 128,
        #learning_rate: float = 0.03,
        #decay: float = 1.0,
        #label_smoothing: float = 0.1,

        self.model = model  #type:Model
        self.dataset = self.model.dataset
        self.batch_size = hyperparameters[BATCH_SIZE]
        self.label_smoothing = hyperparameters[LABEL_SMOOTHING]
        self.learning_rate = hyperparameters[LEARNING_RATE]
        self.decay = hyperparameters[DECAY]
        self.epochs = hyperparameters[EPOCHS]
        self.verbose = verbose

        self.loss = torch.nn.BCELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)  # we only support ADAM for BCE
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.decay)

        # create the evaluator to use between epochs
        self.evaluator = Evaluator(self.model)

    def train(self,
              train_samples: np.array,
              save_path: str = None,
              evaluate_every:int =-1,
              valid_samples:np.array = None):

        print("Training the TuckER model...")

        training_samples = np.vstack((train_samples,
                                      self.dataset.invert_samples(train_samples)))

        er_vocab = self.extract_er_vocab(training_samples)
        er_vocab_pairs = list(er_vocab.keys())

        self.model.cuda()

        for e in range(1, self.epochs+1):
            self.epoch(er_vocab=er_vocab, er_vocab_pairs=er_vocab_pairs, batch_size=self.batch_size)

            if evaluate_every > 0 and valid_samples is not None and e % evaluate_every == 0:
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

    def extract_er_vocab(self, samples):
        er_vocab = defaultdict(list)
        for triple in samples:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def extract_batch(self, er_vocab, er_vocab_pairs, batch_start, batch_size):
        batch = er_vocab_pairs[batch_start: min(batch_start+batch_size, len(er_vocab_pairs))]

        targets = np.zeros((len(batch), self.dataset.num_entities))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        if self.label_smoothing:
            targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))

        return torch.tensor(np.array(batch)).cuda(), torch.FloatTensor(targets).cuda()

    def epoch(self,
              er_vocab,
              er_vocab_pairs,
              batch_size: int):

        np.random.shuffle(er_vocab_pairs)
        self.model.train()

        with tqdm.tqdm(total=len(er_vocab_pairs), unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            batch_start = 0

            while batch_start < len(er_vocab_pairs):
                batch, targets = self.extract_batch(er_vocab=er_vocab,
                                                    er_vocab_pairs=er_vocab_pairs,
                                                    batch_start=batch_start,
                                                    batch_size=batch_size)
                l = self.step_on_batch(batch, targets)
                batch_start+=batch_size
                bar.update(batch_size)
                bar.set_postfix(loss=str(round(l.item(), 6)))

            if self.decay:
                self.scheduler.step()


    def step_on_batch(self, batch, targets):
        self.optimizer.zero_grad()

        predictions = self.model.forward(batch)
        loss = self.loss(predictions, targets)
        loss.backward()
        self.optimizer.step()

        return loss

class KelpieBCEOptimizer(BCEOptimizer):
    def __init__(self,
                 model:Model,
                 hyperparameters: dict,
                 verbose: bool = True):

        super(KelpieBCEOptimizer, self).__init__(model=model,
                                                 hyperparameters=hyperparameters,
                                                 verbose=verbose)

        self.optimizer = optim.Adam(params=self.model.parameters())

    # Override
    def epoch(self,
              er_vocab,
              er_vocab_pairs,
              batch_size: int):

        np.random.shuffle(er_vocab_pairs)
        self.model.train()

        with tqdm.tqdm(total=len(er_vocab_pairs), unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < len(er_vocab_pairs):
                batch, targets = self.extract_batch(er_vocab=er_vocab,
                                                    er_vocab_pairs=er_vocab_pairs,
                                                    batch_start=batch_start,
                                                    batch_size=batch_size)
                l = self.step_on_batch(batch, targets)

                # THIS IS THE ONE DIFFERENCE FROM THE ORIGINAL OPTIMIZER.
                # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES IN THE KELPIE ENTITY EMBEDDING
                # TO THE MATRIX CONTAINING ALL THE EMBEDDINGS
                self.model.update_embeddings()

                batch_start+=batch_size
                bar.update(batch_size)
                bar.set_postfix(loss=str(round(l.item(), 6)))

            if self.decay:
                self.scheduler.step()
import numpy as np
import torch
from tqdm import trange
from tqdm.auto import tqdm
from torch import optim
import torch.nn.functional as F

from kelpie.evaluation import Evaluator
from kelpie.models.hake.model import Hake


class HakeOptimizer:

    def __init__(self,
                 model: Hake,
                 optimizer_name: str = "Adam",
                 learning_rate: float = 0.0001,
                 no_decay: bool = False,
                 max_steps: int = 100000,
                 adversarial_temperature: float = 1.0,
                 save_path: str = None):

        self.model = model
        self.learning_rate = learning_rate
        self.no_decay = no_decay
        self.max_steps = max_steps
        self.adversarial_temperature = adversarial_temperature
        self.save_path = save_path

        # build all the supported optimizers using the passed params
        supported_optimizers = {
            'Adagrad': optim.Adagrad(params=self.model.parameters(), lr=learning_rate),
            'Adam': optim.Adam(params=self.model.parameters(), lr=learning_rate),
            'SGD': optim.SGD(params=self.model.parameters(), lr=learning_rate)
        }
        self.optimizer_name = optimizer_name

        # choose the Torch Optimizer object to use, based on the passed name
        self.optimizer = supported_optimizers[self.optimizer_name]

        # create the evaluator to use between epochs
        self.evaluator = Evaluator(self.model)


    def train(self,
            train_iterator,
            init_step: int = 0,
            evaluate_every: int = -1,
            valid_samples: np.array = None):

        warm_up_steps = self.max_steps // 2
        current_learning_rate = self.learning_rate

        # Training Loop
        for step in range(init_step, self.max_steps):

            loss = None
            if((self.model.num_entities * 2)%self.model.batch_size == 0):
                actual_steps = (self.model.num_entities * 2) // self.model.batch_size
            else:
                actual_steps = ((self.model.num_entities * 2) // self.model.batch_size) + 1

            #with tqdm.tqdm(total=actual_steps) as bar:
                #bar.set_description('train loss: epoch #'+str(step))   , bar_format='{l_bar}{bar}|{elapsed}, {postfix}'
            #with tqdm(range(actual_steps), desc='epoch #'+str(step)) as bar:
            with trange(actual_steps) as bar:
                bar.set_description('epoch #' + str(step))
                bar.set_postfix(loss="...")
                for i in bar:
                    loss = self.train_step(train_iterator)
                    np.random.seed()    #resets np.random seed
                    #bar.update(i)
                bar.set_postfix(loss="{:.4f}".format(loss.item()))

                #bar.close()
                #print("loss: "+str(loss[0]))

            if step >= warm_up_steps:
                if not self.no_decay:
                    current_learning_rate = current_learning_rate / 10
                    print('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                supported_optimizers = {
                    'Adagrad': optim.Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             lr=current_learning_rate),
                    'Adam': optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                       lr=current_learning_rate),
                    'SGD': optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=current_learning_rate)
                }
                self.optimizer = supported_optimizers[self.optimizer_name]
                warm_up_steps = warm_up_steps * 3

            if evaluate_every > 0 and valid_samples is not None and (step + 1) % evaluate_every == 0:
                mrr, h1 = self.evaluator.eval(samples=valid_samples, write_output=False)

                print("\tValidation Hits@1: %f" % h1)
                print("\tValidation Mean Reciprocal Rank': %f" % mrr)

                if self.save_path is not None:
                    print("\t saving model...")
                    torch.save(self.model.state_dict(), self.save_path)
                print("\t done.")


    def train_step(self, train_iterator):
        '''
        A single train step. Apply back-propagation and return the loss
        '''

        self.model.train()

        self.optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
                        # gamma - dr(h, t)
        negative_score = self.model((positive_sample, negative_sample), batch_type=batch_type)

                        # p(h,r,t)                      # alpha
        negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                            # E log sigma (dr(h, t) - gamma)
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score = self.model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        self.optimizer.step()

        return loss


class KelpieHakeOptimizer(HakeOptimizer):
    def __init__(self,
                 model: Hake,
                 optimizer_name: str = "Adam",
                 learning_rate: float = 0.0001,
                 no_decay: bool = False,
                 max_steps: int = 100000,
                 adversarial_temperature: float = 1.0,
                 save_path: str = None):

        super(KelpieHakeOptimizer, self).__init__(model=model,
                                                     optimizer_name=optimizer_name,
                                                     learning_rate=learning_rate,
                                                     no_decay=no_decay,
                                                     max_steps=max_steps,
                                                     adversarial_temperature=adversarial_temperature,
                                                     save_path=save_path)

    def train_step(self, train_iterator):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        self.model.train()

        self.optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
        # gamma - dr(h, t)
        negative_score = self.model((positive_sample, negative_sample), batch_type=batch_type)

        # p(h,r,t)                      # alpha
        negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                          # E log sigma (dr(h, t) - gamma)
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score = self.model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        self.optimizer.step()

        # THIS IS THE ONE DIFFERENCE FROM THE ORIGINAL OPTIMIZER.
        # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES IN THE KELPIE ENTITY EMBEDDING
        # TO THE MATRIX CONTAINING ALL THE EMBEDDINGS
        self.model.update_embeddings()

        return loss

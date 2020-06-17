from typing import Tuple, Any

import torch
from torch import nn
import numpy as np
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

from kelpie.dataset import Dataset
from kelpie.kelpie_dataset import KelpieDataset
from kelpie.model import Model


class TuckER(Model, nn.Module):
    """
    """


    def __init__(self,
                 dataset: Dataset,
                 entity_dimension: int,
                 relation_dimension:int,
                 input_dropout: float,
                 hidden_dropout1: float,
                 hidden_dropout2: float,
                 init_random = True):
        """
        """

        # note: the init_random parameter is important because when initializing a KelpieTuckER,
        #       self.entity_embeddings and self.relation_embeddings must not be initialized as Parameters!

        # initialize this object both as a Model and as a nn.Module
        Model.__init__(self, dataset)
        nn.Module.__init__(self)

        self.dataset = dataset
        self.num_entities = dataset.num_entities     # number of entities in dataset
        self.num_relations = dataset.num_relations   # number of relations in dataset
        self.entity_dimension = entity_dimension     # entity embedding dimension
        self.relation_dimension = relation_dimension # relation embedding dimension

        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(hidden_dropout2)
        self.batch_norm1 = torch.nn.BatchNorm1d(entity_dimension)
        self.batch_norm2 = torch.nn.BatchNorm1d(entity_dimension)

        # create the embeddings for entities and relations as Parameters.
        # We do not use the torch.Embeddings module here in order to keep the code uniform to the KelpieTuckER model,
        # (on which torch.Embeddings can not be used as they do not allow the post-training mechanism).
        # We have verified that this does not affect performances in any way.
        if init_random:
            self.entity_embeddings = Parameter(torch.empty(self.num_entities, self.entity_dimension).cuda(), requires_grad=True)
            self.relation_embeddings = Parameter(torch.empty(self.num_relations, self.relation_dimension).cuda(), requires_grad=True)
            self.core_tensor = Parameter(torch.tensor(np.random.uniform(-1, 1, (self.relation_dimension, self.entity_dimension, self.entity_dimension)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
            
            xavier_normal_(self.entity_embeddings.weight.data)
            xavier_normal_(self.relation_embeddings.weight.data)

    def score(self, samples: np.array) -> np.array:
        """
        """

    def forward(self, samples: np.array):
        """
        """
        head_indexes = samples[:, 0]
        head_embeddings = self.entity_embeddings[head_indexes]
        relation_indexes = samples[:, 1]
        relation_embeddings = self.relation_embeddings[relation_indexes]
        tail_embeddings = self.entity_embeddings.weight
        core_tensor_reshaped = self.core_tensor.view(self.relation_dimension, -1)

        first_multiplication = torch.mm(relation_embeddings, core_tensor_reshaped)
        first_multiplication_reshaped = first_multiplication.view(-1, self.entity_dimension, self.entity_dimension)
        first_multiplication_reshaped = self.hidden_dropout1(first_multiplication_reshaped)

        head_embeddings = self.batch_norm1(head_embeddings)
        head_embeddings = self.input_dropout(head_embeddings)
        head_embeddings_reshaped = head_embeddings.view(-1, 1, self.entity_dimension)

        second_multiplication = torch.bmm(head_embeddings_reshaped, first_multiplication_reshaped) 
        second_multiplication_reshaped = second_multiplication.view(-1, self.entity_dimension)      
        second_multiplication_reshaped = self.batch_norm2(second_multiplication_reshaped)
        second_multiplication_reshaped = self.hidden_dropout2(second_multiplication_reshaped)
        
        tail_embeddings_transposed = tail_embeddings.transpose(1,0)

        result = torch.mm(second_multiplication_reshaped, tail_embeddings_transposed)
        predictions = torch.sigmoid(result)

        return predictions
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
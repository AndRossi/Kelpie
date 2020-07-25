import torch
import numpy
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.functional import F

import numpy as np
import torch
from torch import nn


class Permutator(nn.Module):

    def __init__(self,
                #  embed_dim: int,
                 num_perm: int = 1,
                 reshpd_mtx_h: int = 20,
                 reshpd_mtx_w: int = 10,
                 device: str = '-1'):

        # self.embed_dim = embed_dim
        self.num_perm = num_perm
        self.reshpd_mtx_h = reshpd_mtx_h
        self.reshpd_mtx_w = reshpd_mtx_w
        self.embed_dim = (reshpd_mtx_h * reshpd_mtx_w) # follows the paper formula

        if device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
    

    def chequer_perm(self):
        """
        Function to generate the chequer permutation required for InteractE model

        Parameters
        ----------
        Returns
        -------
        """
        # ent_perms and rel_perms are lists of permutations
        ent_perms = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.num_perm)])
        rel_perms = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.num_perm)])

        comb_idx = [] # matrice da costruire
        for k in range(self.num_perm): 
            temp = [] # riga corrente della matrice
            ent_idx, rel_idx = 0, 0

            # ciclo sulle righe della matriche risultante
            for i in range(self.reshpd_mtx_h):
                # ciclo sulle colonne della matriche risultante
                for j in range(self.reshpd_mtx_w):
                    # if k is even
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perms[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perms[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perms[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perms[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perms[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perms[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perms[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perms[k, rel_idx] + self.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm


embed_dim = 300
k_h = 20
k_w = 10
inp_drop_p = 0.5
hid_drop_p = 0.5
feat_drop_p = 0.5
num_perm = 3
kernel_size = 9
num_filt_conv = 96
init_random = True		#? forse non sarà usato
init_size = 1e-3 #? forse non sarà usato
strategy ='one_to_n'


def circular_padding_chw (batch, padding):
    upper_pad	= batch[..., -padding:, :]
    lower_pad	= batch[..., :padding, :]
    temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

    left_pad	= temp[..., -padding:]
    right_pad	= temp[..., :padding]
    padded		= torch.cat([left_pad, temp, right_pad], dim=3)
    return padded


dataset = np.random.rand(200,3)
num_entities = len(dataset[:, 0])		# number of entities in dataset
num_relations = len(dataset[:, 1])		# number of relations in dataset


# Subject and relationship embeddings, xavier_normal_ distributes 
# the embeddings weight values by the said distribution
ent_embed = torch.nn.Embedding(num_entities, embed_dim, padding_idx=None) 
xavier_normal_(ent_embed.weight)
# num_relation is x2 since we need to embed direct and inverse relationships
rel_embed = torch.nn.Embedding(num_relations*2, embed_dim, padding_idx=None)
xavier_normal_(rel_embed.weight)

# Binary Cross Entropy Loss
bceloss = torch.nn.BCELoss()

# Dropout regularization (default p = 0.5)
inp_drop = torch.nn.Dropout(inp_drop_p)
hidden_drop = torch.nn.Dropout(hid_drop_p)
# Dropout regularization on embedding matrix (default p = 0.5)
feature_map_drop = torch.nn.Dropout2d(feat_drop_p)

# Embedding matrix normalization
bn0 = torch.nn.BatchNorm2d(num_perm)

flat_sz_h = k_h
flat_sz_w = 2*k_w
padding = 0

# Conv layer normalization 
bn1 = torch.nn.BatchNorm2d(num_filt_conv * num_perm)

# Flattened embedding matrix size
flat_sz = flat_sz_h * flat_sz_w * num_filt_conv * num_perm

# Normalization 
bn2 = torch.nn.BatchNorm1d(embed_dim)

# Matrix flattening
fc = torch.nn.Linear(flat_sz, embed_dim)

# Chequered permutation
chequer_perm = Permutator(num_perm).chequer_perm()

# Bias definition
bias = torch.zeros(num_entities)

# Kernel filter definition
conv_filt = torch.zeros(num_filt_conv, 1, kernel_size, kernel_size)
xavier_normal_(conv_filt)


def score(samples: numpy.array) -> numpy.array:

    sub_samples = torch.LongTensor(np.int32(samples[:, 0]))
    rel_samples = torch.LongTensor(np.int32(samples[:, 1]))
    
    #score = sigmoid(torch.cat(ReLU(conv_circ(embedding_matrix, kernel_tensor)))weights)*embedding_o
    sub_emb	= ent_embed(sub_samples)	# Embeds the subject tensor
    rel_emb	= ent_embed(rel_samples)	# Embeds the relationship tensor
    
    comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
    # to access local variable.
    matrix_chequer_perm = comb_emb[:, chequer_perm]
    # matrix reshaped
    stack_inp = matrix_chequer_perm.reshape((-1, num_perm, 2*k_w, k_h))
    stack_inp = bn0(stack_inp)  # Normalizes
    x = inp_drop(stack_inp)	# Regularizes with dropout
    # Circular convolution
    x = circular_padding_chw(x, kernel_size//2)	# Defines the kernel for the circular conv
    x = F.conv2d(x, conv_filt.repeat(num_perm, 1, 1, 1), padding=padding, groups=num_perm) # Circular conv
    x = bn1(x)	# Normalizes
    x = F.relu(x)
    x = feature_map_drop(x)	# Regularizes with dropout
    x = x.view(-1, flat_sz)
    x = fc(x)
    x = hidden_drop(x)	# Regularizes with dropout
    x = bn2(x)	# Normalizes
    x = F.relu(x)
    
    # if strategy == 'one_to_n':
    #     x = torch.mm(x, ent_embed.weight.transpose(1,0))
    #     x += bias.expand_as(x)rt
    # else:
    #     x = torch.mul(x.unsqueeze(1), ent_embed(neg_ents)).sum(dim=-1)
    #     x += bias[neg_ents]

    pred = torch.sigmoid(x)

    return pred

print(score(dataset).shape)
print(score(dataset))  

# import numpy as np

# sorted_samples = np.array([[4,7,6],[1,2,5],[9,3,8]])
# print(sorted_samples)

# tail_id = np.array([4,4,6])

# print(np.where(sorted_samples[0]==tail_id[0].item()))
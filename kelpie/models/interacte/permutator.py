import numpy as np
import torch
from torch import nn


class Permutator(nn.Module):

    def __init__(self,
                    # embed_dim: int,
                    num_perm: int = 1,
                    mtx_h: int = 20,
                    mtx_w: int = 10): #,

        # self.embed_dim = embed_dim
        self.num_perm = num_perm
        self.mtx_h = mtx_h
        self.mtx_w = mtx_w
        self.embed_dim = (mtx_h * mtx_w)	# follows the paper formula
        
        self.device = torch.device('cuda')


    def chequer_perm(self):
        """
        Function to generate the chequer permutation required for InteractE model.

        Parameters
        ----------
        Returns
        -------
        A tensor (k x d), where k is the number of permutations and 
        d is the dimension of chequer combination of embeddings.

        """
        # ent_perms and rel_perms are lists of permutations
        ent_perms = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.num_perm)])
        rel_perms = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.num_perm)])

        comb_idx = [] # matrix of size k x 
        for k in range(self.num_perm): 
            temp = [] # current row of the matrix
            ent_idx, rel_idx = 0, 0

            # for all row in the matrix
            for i in range(self.mtx_h):
                # for all column in the matrix
                for j in range(self.mtx_w):
                    # if k is even
                    if k % 2 == 0:
                        # if i is even
                        if i % 2 == 0:
                            temp.append(ent_perms[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perms[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                        # if i is odd
                        else:
                            temp.append(rel_perms[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perms[k, ent_idx])
                            ent_idx += 1
                    # if k is odd
                    else:
                        # if i is even
                        if i % 2 == 0:
                            temp.append(rel_perms[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perms[k, ent_idx])
                            ent_idx += 1
                        # if i is odd
                        else:
                            temp.append(ent_perms[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perms[k, rel_idx] + self.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm
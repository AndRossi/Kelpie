from helper import *
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from data_loader import *
from model import *


class Permutator():
	
    def __init(self, params):
        
        self.p = params
    
    def get_chequer_perm(self):
			"""
			Function to generate the chequer permutation required for InteractE model

			Parameters
			----------
			
			Returns
			-------
			
			"""
            # embed_dim is the embedding dimension for ent end rel embeddings; None by default 
            #   and not considered (?) if a k_h and k_w are defined
            # p.perm represents the number of permutations to execute
			ent_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])
			rel_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])

			comb_idx = []
			for k in range(self.p.perm): 
				temp = []
				ent_idx, rel_idx = 0, 0

				for i in range(self.p.k_h):
					for j in range(self.p.k_w):
						#if k is even
                        if k % 2 == 0:
							if i % 2 == 0:
								temp.append(ent_perm[k, ent_idx])
                                ent_idx += 1
								temp.append(rel_perm[k, rel_idx]+self.p.embed_dim)
                                rel_idx += 1
							else:
								temp.append(rel_perm[k, rel_idx]+self.p.embed_dim)
                                rel_idx += 1
								temp.append(ent_perm[k, ent_idx])
                                ent_idx += 1
						else:
							if i % 2 == 0:
								temp.append(rel_perm[k, rel_idx]+self.p.embed_dim)
                                rel_idx += 1
								temp.append(ent_perm[k, ent_idx])
                                ent_idx += 1
							else:
								temp.append(ent_perm[k, ent_idx])
                                ent_idx += 1
								temp.append(rel_perm[k, rel_idx]+self.p.embed_dim)
                                rel_idx += 1

				comb_idx.append(temp)

			chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
			return chequer_perm
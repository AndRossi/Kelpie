import numpy as np
import torch
from torch import nn


class Permutator(nn.Module):

    def __init__(self,
            	 # embed_dim: int,
            	 permutations: int = 1,
            	 k_h: int = 20,
            	 k_w: int = 10,
				 device: str = '-1'):

		# self.embed_dim = embed_dim
		self.permutations = permutations
		self.k_h = k_h
		self.k_w = k_w
		self.embed_dim = (k_h * k_w) / 2	# follows the paper formula

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
		ent_perm  = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.permutations)])
		rel_perm  = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.permutations)])

		comb_idx = [] # matrice da costruire
		for k in range(self.permutations): 
			temp = [] # riga corrente della matrice
			ent_idx, rel_idx = 0, 0

			# ciclo sulle righe della matriche risultante
			for i in range(self.k_h):
				# ciclo sulle colonne della matriche risultante
				for j in range(self.k_w):
					# if k is even
					if k % 2 == 0:
						if i % 2 == 0:
							temp.append(ent_perm[k, ent_idx])
							ent_idx += 1
							temp.append(rel_perm[k, rel_idx] + self.embed_dim)
							rel_idx += 1
						else:
							temp.append(rel_perm[k, rel_idx] + self.embed_dim)
							rel_idx += 1
							temp.append(ent_perm[k, ent_idx])
							ent_idx += 1
					else:
						if i % 2 == 0:
							temp.append(rel_perm[k, rel_idx] + self.embed_dim)
							rel_idx += 1
							temp.append(ent_perm[k, ent_idx])
							ent_idx += 1
						else:
							temp.append(ent_perm[k, ent_idx])
							ent_idx += 1
							temp.append(rel_perm[k, rel_idx] + self.embed_dim)
							rel_idx += 1

			comb_idx.append(temp)

		# valutare se bisogna modificare la costruzione della chequer_perm (?)
		chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
		return chequer_perm
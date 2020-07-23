from typing import Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from kelpie.dataset import Dataset
from kelpie.kelpie_dataset import KelpieDataset
from kelpie.model import Model

from helper import *

class InteractE(Model, torch.nn.Module):
	"""
	Proposed method in the paper. Refer Section 6 of the paper for mode details 

	Parameters
	----------
	params:        	Hyperparameters of the model
	chequer_perm:   Reshaping to be used by the model
	
	Returns
	-------
	The InteractE model instance
		
	"""
	def __init__(self, 
		chequer_perm,
		dataset: Dataset,
		embed_dim: int,
		k_h: int = 20,
		k_w: int = 10,
		inp_drop_p: float = 0.5,
		hid_drop_p: float = 0.5,
		feat_drop_p: float = 0.5,
		num_perm: int = 1,
		kernel_size: int = 9,
		num_filt_conv: int = 96,
		init_random = True,		#? forse non sarà usato
		init_size: float = 1e-3,#? forse non sarà usato
		strategy: str='one_to_n'):

        # initialize this object both as a Model and as a nn.Module
		Model.__init(self, dataset)
		nn.Module.__init__(self)

		self.dataset = dataset
		self.num_entities = dataset.num_entities		# number of entities in dataset
		self.num_relations = dataset.num_relations		# number of relations in dataset
		self.embed_dim = embed_dim						# embedding dimension
		self.num_perm = num_perm						# number of permutation
		self.kernel_size = kernel_size

		self.strategy=strategy
		
		# Inverted Entities
		self.neg_ents
		
		# Subject and relationship embeddings, xavier_normal_ distributes 
		# the embeddings weight values by the said distribution
		self.ent_embed = torch.nn.Embedding(self.num_entities, embed_dim, padding_idx=None) 
		xavier_normal_(self.ent_embed.weight)
		# num_relation is x2 since we need to embed direct and inverse relationships
		self.rel_embed = torch.nn.Embedding(self.num_relations*2, embed_dim, padding_idx=None)
		xavier_normal_(self.rel_embed.weight)
		
		# Binary Cross Entropy Loss
		self.bceloss = torch.nn.BCELoss()

		# Dropout regularization (default p = 0.5)
		self.inp_drop = torch.nn.Dropout(inp_drop_p)
		self.hidden_drop = torch.nn.Dropout(hid_drop_p)
		# Dropout regularization on embedding matrix (default p = 0.5)
		self.feature_map_drop = torch.nn.Dropout2d(feat_drop_p)
		
		# Embedding matrix normalization
		self.bn0 = torch.nn.BatchNorm2d(num_perm)

		flat_sz_h = k_h
		flat_sz_w = 2*k_w
		self.padding = 0

		# Conv layer normalization 
		self.bn1 = torch.nn.BatchNorm2d(num_filt_conv * num_perm)
		
		# Flattened embedding matrix size
		self.flat_sz = flat_sz_h * flat_sz_w * num_filt_conv * num_perm

		# Normalization 
		self.bn2 = torch.nn.BatchNorm1d(embed_dim)

		# Matrix flattening
		self.fc = torch.nn.Linear(self.flat_sz, embed_dim)
		
		# Chequered permutation
		self.chequer_perm = chequer_perm

		# Bias definition
		self.register_parameter('bias', Parameter(torch.zeros(self.num_entities)))

		# Kernel filter definition
		self.register_parameter('conv_filt', Parameter(torch.zeros(num_filt_conv, 1, kernel_size, kernel_size)))
		xavier_normal_(self.conv_filt)


	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos = true_label[0]; 
		label_neg = true_label[1:]
		loss = self.bceloss(pred, true_label)
		return loss


	def score(self, samples: numpy.array) -> numpy.array:
		"""
			This method computes and returns the plausibility scores for a collection of samples.

			:param samples: a numpy array containing all the samples to score
			:return: the computed scores, as a numpy array
		"""
	
		sub_samples = torch.LongTensor(np.int32(samples[:, 0]))
		rel_samples = torch.LongTensor(np.int32(samples[:, 1]))
		
        #score = sigmoid(torch.cat(ReLU(conv_circ(embedding_matrix, kernel_tensor)))weights)*embedding_o
		sub_emb	= self.ent_embed(sub_samples)	# Embeds the subject tensor
		rel_emb	= self.ent_embed(rel_samples)	# Embeds the relationship tensor
		
		comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
		# self to access local variable.
		matrix_chequer_perm = comb_emb[:, self.chequer_perm]
		# matrix reshaped
		stack_inp = matrix_chequer_perm.reshape((-1, self.num_perm, 2*k_w, k_h))
		stack_inp = self.bn0(stack_inp)  # Normalizes
		x = self.inp_drop(stack_inp)	# Regularizes with dropout
		# Circular convolution
		x = self.circular_padding_chw(x, self.kernel_size//2)	# Defines the kernel for the circular conv
		x = F.conv2d(x, self.conv_filt.repeat(self.num_perm, 1, 1, 1), padding=self.padding, groups=self.num_perm) # Circular conv
		x = self.bn1(x)	# Normalizes
		x = F.relu(x)
		x = self.feature_map_drop(x)	# Regularizes with dropout
		x = x.view(-1, self.flat_sz)
		x = self.fc(x)
		x = self.hidden_drop(x)	# Regularizes with dropout
		x = self.bn2(x)	# Normalizes
		x = F.relu(x)
		
		if self.strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(self.neg_ents)).sum(dim=-1)
			x += self.bias[self.neg_ents]

		pred = torch.sigmoid(x)

		return pred.numpy()


	# Circular padding definition
	def circular_padding_chw(self, batch, padding):
		upper_pad	= batch[..., -padding:, :]
		lower_pad	= batch[..., :padding, :]
		temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

		left_pad	= temp[..., -padding:]
		right_pad	= temp[..., :padding]
		padded		= torch.cat([left_pad, temp, right_pad], dim=3)
		return padded


	# Forwards data throughout the network
	def forward(self, samples: numpy.array) -> numpy.array:
		"""
			This method performs forward propagation for a collection of samples.
			This method is only used in training, when an Optimizer calls it passing the current batch of samples.

			This method returns all the items needed by the Optimizer to perform gradient descent in this training step.
			Such items heavily depend on the specific Model implementation;
			they usually include the scores for the samples (in a form usable by the ML framework, e.g. torch.Tensors)
			but may also include other stuff (e.g. the involved embeddings themselves, that the Optimizer
			may use to compute regularization factors)

			:param samples: a numpy array containing all the samples to perform forward propagation on
		"""
	
		sub_samples = torch.LongTensor(np.int32(samples[:, 0]))
		rel_samples = torch.LongTensor(np.int32(samples[:, 1]))
		
        #score = sigmoid(torch.cat(ReLU(conv_circ(embedding_matrix, kernel_tensor)))weights)*embedding_o
		sub_emb	= self.ent_embed(sub_samples)	# Embeds the subject tensor
		rel_emb	= self.ent_embed(rel_samples)	# Embeds the relationship tensor
		
		comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
		# self to access local variable.
		matrix_chequer_perm = comb_emb[:, self.chequer_perm]
		# matrix reshaped
		stack_inp = matrix_chequer_perm.reshape((-1, self.num_perm, 2*k_w, k_h))
		stack_inp = self.bn0(stack_inp)  # Normalizes
		x = self.inp_drop(stack_inp)	# Regularizes with dropout
		# Circular convolution
		x = self.circular_padding_chw(x, self.kernel_size//2)	# Defines the kernel for the circular conv
		x = F.conv2d(x, self.conv_filt.repeat(self.num_perm, 1, 1, 1), padding=self.padding, groups=self.num_perm) # Circular conv
		x = self.bn1(x)	# Normalizes
		x = F.relu(x)
		x = self.feature_map_drop(x)	# Regularizes with dropout
		x = x.view(-1, self.flat_sz)
		x = self.fc(x)
		x = self.hidden_drop(x)	# Regularizes with dropout
		x = self.bn2(x)	# Normalizes
		x = F.relu(x)
		
		if self.strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(self.neg_ents)).sum(dim=-1)
			x += self.bias[self.neg_ents]

		pred = torch.sigmoid(x)

		return pred.numpy()


	def predict_samples(self, samples: numpy.array) -> Tuple[Any, Any, Any]:
		"""
			This method performs prediction on a collection of samples, and returns the corresponding
			scores, ranks and prediction lists.

			All the passed samples must be DIRECT samples in the original dataset.
			(if the Model supports inverse samples as well,
			it should invert the passed samples while running this method)

			:param samples: the direct samples to predict, in numpy array format
			:return: this method returns three lists:
						- the list of scores for the passed samples,
									OR IF THE MODEL SUPPORTS INVERSE FACTS
							the list of couples <direct sample score, inverse sample score>,
							where the i-th score refers to the i-th sample in the input samples.

						- the list of couples (head rank, tail rank)
							where the i-th couple refers to the i-th sample in the input samples.

						- the list of couples (head_predictions, tail_predictions)
							where the i-th couple refers to the i-th sample in the input samples.
							The head_predictions and tail_predictions for each sample
							are numpy arrays containing all the predicted heads and tails respectively for that sample.
		"""
		pass


	def predict_sample(self, sample: numpy.array) -> Tuple[Any, Any, Any]:
		"""
			This method performs prediction on one (direct) sample, and returns the corresponding
			score, ranks and prediction lists.

			:param sample: the sample to predict, as a numpy array.
			:return: this method returns 3 items:
					- the sample score
								OR IF THE MODEL SUPPORTS INVERSE FACTS
						the scores of the sample and of its inverse

					- a couple containing the head rank and the tail rank

					- a couple containing the head_predictions and tail_predictions numpy arrays;
						> head_predictions contains all entities predicted as heads, sorted by decreasing plausibility
						[NB: the target head will be in this numpy array in position head_rank-1]
						> tail_predictions contains all entities predicted as tails, sorted by decreasing plausibility
						[NB: the target tail will be in this numpy array in position tail_rank-1]
		"""

		assert sample[1] < self.dataset.num_direct_relations

		scores, ranks, predictions = self.predict_samples(numpy.array([sample]))
		return scores[0], ranks[0], predictions[0]


class KelpieInteractE(InteractE):
	# Constructor
	def __init__(
		self,
		dataset: KelpieDataset,
		model: InteractE,
		params):

		#Parameters
		#----------
		#params:        	Hyperparameters of the model
		#chequer_perm:   	Reshaping to be used by the model

		InteractE.__init__(self,
			dataset=dataset,
			dimension=model.dimension,
			init_random=False,
			init_size=init_size)
		
		self.model = model

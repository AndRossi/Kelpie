from typing import Tuple

import torch
from torch import nn

class Regularizer(nn.Module):

    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class L2(Regularizer):
    def __init__(self, weight: float):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        l = 0
        for factor in factors:
            l += torch.mean(factor ** 2)
        l = l*self.weight/len(factors)
        return l


class N2(Regularizer):
    def __init__(self, weight: float):
        super(N2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.norm(f, 2, 1) ** 3
            )
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    # factors is 3 matrices that will then be used for regularization
    #       matrix 1: for each training fact
    #                           get the head entity real and imaginary components (as separate vectors)
    #                           square each of their elements
    #                           sum the resulting vectors
    #                           squareroot the elements of the resulting vector
    #       matrix 2: for each training fact,
    #                           get the relation real and imaginary components (as separate vectors)
    #                           square each of their elements
    #                           sum the resulting vectors
    #                           squareroot the elements of the resulting vector
    #       matrix 3: for each training fact
    #                           get the tail entity real and imaginary components (as separate vectors)
    #                           square each of their elements
    #                           sum the resulting vectors
    #                           squareroot the elements of the resulting vector
    #
    # this method takes those factors (which are matrices),
    #       raises their elements to their 3d power,
    #       sums their values
    # sums the obtained 3 values
    # divides the obtained sum by the shape of the first factor, that is, the number of training elements

    # THE GOAL OF THIS REGULARIZATION IS TO AVOID HAVING TOO LARGE VALUES IN OUR EMBEDDINGS
    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]

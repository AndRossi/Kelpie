from enum import Enum

import numpy as np
import torch


class BatchType(Enum):
    HEAD_BATCH = 0
    TAIL_BATCH = 1
    SINGLE = 2

def get_test_step_samples(self, pos_triple):
    """
    Returns a positive sample and `self.neg_size` negative samples.
    """
    head, rel, tail = pos_triple

    subsampling_weight = hr_freq[(head, rel)] + tr_freq[(tail, rel)]
    subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

    neg_triples = []
    neg_size = 0

    while neg_size < self.neg_size:
        neg_triples_tmp = np.random.randint(num_entity, size=neg_size * 2)
        if self.batch_type == BatchType.HEAD_BATCH:
            mask = np.in1d(
                neg_triples_tmp,
                tr_map[(tail, rel)],
                assume_unique=True,
                invert=True
            )
        elif self.batch_type == BatchType.TAIL_BATCH:
            mask = np.in1d(
                neg_triples_tmp,
                hr_map[(head, rel)],
                assume_unique=True,
                invert=True
            )
        else:
            raise ValueError('Invalid BatchType: {}'.format(self.batch_type))

        neg_triples_tmp = neg_triples_tmp[mask]
        neg_triples.append(neg_triples_tmp)
        neg_size += neg_triples_tmp.size

    neg_triples = np.concatenate(neg_triples)[:self.neg_size]

    pos_triple = torch.LongTensor(pos_triple)
    neg_triples = torch.from_numpy(neg_triples)

    return pos_triple, neg_triples, subsampling_weight, BatchType.TAIL_BATCH

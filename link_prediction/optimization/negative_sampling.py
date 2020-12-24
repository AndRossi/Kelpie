import random

import numpy

from dataset import Dataset

class CorruptionEngine:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.all_entities = list(range(dataset.num_entities))

    def corrupt_samples(self,
                        positive_samples: numpy.array,
                        corrupt_rate: int):

        out_positive_samples = []
        out_negative_samples = []

        for cur_positive_sample in positive_samples:

            cur_negative_samples = self.corrupt_sample(cur_positive_sample, corrupt_rate)

            for i in range(corrupt_rate):
                out_positive_samples.append(cur_positive_sample)
                out_negative_samples.append(cur_negative_samples[i])

        return numpy.array(out_positive_samples), numpy.array(out_negative_samples)

    def corrupt_sample(self, positive_sample, corrupt_rate):

        correct_head, correct_rel, correct_tail = positive_sample

        negative_samples = []

        used_corrupted_tails = set()
        used_corrupted_heads = set()

        generated_negative_samples_number = 0
        while generated_negative_samples_number < corrupt_rate:
            tail_corruption = bool(random.getrandbits(1))

            # generate a corrupted sample by replacing the correct_tail with a random entity
            #   - not seen as a correct tail prediction answer in training for the current correct_head and correct_rel
            #   - and not already used for generating a negative sample for this positive sample
            if tail_corruption:
                corrupted_tail = random.choice(self.all_entities)
                while corrupted_tail in self.dataset.train_to_filter[(correct_head, correct_rel)] and corrupted_tail not in used_corrupted_tails:
                    corrupted_tail = random.choice(self.all_entities)
                used_corrupted_tails.add(corrupted_tail)
                negative_samples.append((correct_head, correct_rel, corrupted_tail))

            # generate a corrupted sample by replacing the correct_head with a random entity not in
            #   - not seen as a correct head prediction answer in training for the current correct_rel and correct_tail
            #   - and not already used for generating a negative sample for this positive sample
            else:
                corrupted_head = random.choice(self.all_entities)
                while corrupted_head in self.dataset.train_to_filter[(correct_tail, correct_rel + self.dataset.num_direct_relations)] and corrupted_head not in used_corrupted_heads:
                    corrupted_head = random.choice(self.all_entities)
                used_corrupted_heads.add(corrupted_head)
                negative_samples.append((corrupted_head, correct_rel, correct_tail))

            generated_negative_samples_number += 1

        return negative_samples
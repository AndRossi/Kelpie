import html

from link_prediction.models.transe import TransE
from link_prediction.models.model import Model, KelpieModel
import numpy as np

class Evaluator:
    def __init__(self, model: Model):
        self.model = model
        self.dataset = model.dataset    # the Dataset may be useful to convert ids to names

    def evaluate(self,
                samples: np.array,
                write_output:bool = False):

        self.model.cuda()

        # if the model is Transe, it uses too much memory to allow computation of all samples altogether
        batch_size = 500
        if len(samples) > batch_size and isinstance(self.model, TransE):
            scores, ranks, predictions = [], [], []
            batch_start = 0
            while batch_start < len(samples):
                cur_batch = samples[batch_start: min(len(samples), batch_start+batch_size)]
                cur_batch_scores, cur_batch_ranks, cur_batch_predictions = self.model.predict_samples(cur_batch)
                scores += cur_batch_scores
                ranks += cur_batch_ranks
                predictions += cur_batch_predictions

                batch_start += batch_size
        else:
            # run prediction on all the samples
            scores, ranks, predictions = self.model.predict_samples(samples)

        all_ranks = []
        for i in range(samples.shape[0]):
            all_ranks.append(ranks[i][0])
            all_ranks.append(ranks[i][1])

        if write_output:
            self._write_output(samples, ranks, predictions)

        return self.mrr(all_ranks), self.hits_at(all_ranks, 1), self.hits_at(all_ranks, 10), self.mr(all_ranks)

    def _write_output(self, samples, ranks, predictions):
        result_lines = []
        detail_lines = []
        for i in range(samples.shape[0]):
            head_id, rel_id, tail_id = samples[i]

            head_rank, tail_rank = ranks[i]
            head_prediction_ids, tail_prediction_ids = predictions[i]

            head_prediction_ids = head_prediction_ids[:head_rank]
            tail_prediction_ids = tail_prediction_ids[:tail_rank]

            head_name = self.dataset.get_name_for_entity_id(head_id)
            rel_name = self.dataset.get_name_for_relation_id(rel_id)
            tail_name = self.dataset.get_name_for_entity_id(tail_id)

            textual_fact_key = ";".join([head_name, rel_name, tail_name])

            result_lines.append(textual_fact_key + ";" + str(head_rank) + ";" + str(tail_rank) + "\n")

            head_prediction_names = [self.dataset.get_name_for_entity_id(x) for x in head_prediction_ids]
            tail_prediction_names = [self.dataset.get_name_for_entity_id(x) for x in tail_prediction_ids]

            detail_lines.append(textual_fact_key + ";predict head;[" + ";".join(head_prediction_names) + "]\n")
            detail_lines.append(textual_fact_key + ";predict tail;[" + ";".join(tail_prediction_names) + "]\n")

        for i in range(len(result_lines)):
            result_lines[i] = html.unescape(result_lines[i])
        for i in range(len(detail_lines)):
            detail_lines[i] = html.unescape(detail_lines[i])

        with open("filtered_ranks.csv", "w") as output_file:
            output_file.writelines(result_lines)
        with open("filtered_details.csv", "w") as output_file:
            output_file.writelines(detail_lines)

    @staticmethod
    def mrr(values):
        mrr = 0.0
        for value in values:
            mrr += 1.0 / float(value)
        mrr = mrr / float(len(values))
        return mrr

    @staticmethod
    def mr(values):
        return np.average(values)

    @staticmethod
    def hits_at(values, k:int):
        hits = 0
        for value in values:
            if value <= k:
                hits += 1
        return float(hits) / float(len(values))


class KelpieEvaluator(Evaluator):

    def __init__(self, model: KelpieModel):
        super().__init__(model)
        self.model = model

    # override
    def evaluate(self,
                 samples: np.array,
                 write_output:bool = False,
                 original_mode:bool = False):

        batch_size = 1000

        # if the model is Transe, it uses too much memory to allow computation of all samples altogether
        if len(samples) > batch_size and isinstance(self.model, TransE):
            scores, ranks, predictions = [], [], []
            batch_start = 0
            while batch_start < len(samples):
                cur_batch = samples[batch_start: min(len(samples), batch_start+batch_size)]
                cur_batch_scores, cur_batch_ranks, cur_batch_predictions = self.model.predict_samples(cur_batch, original_mode)
                scores += cur_batch_scores
                ranks += cur_batch_ranks
                predictions += cur_batch_predictions

                batch_start += batch_size

        else:
            # run prediction on all the samples
            scores, ranks, predictions = self.model.predict_samples(samples, original_mode)

        all_ranks = []
        for i in range(samples.shape[0]):
            all_ranks.append(ranks[i][0])
            all_ranks.append(ranks[i][1])

        if write_output:
            self._write_output(samples, ranks, predictions)

        return self.mrr(all_ranks), self.hits_at(all_ranks, 1), self.hits_at(all_ranks, 10), self.mr(all_ranks)

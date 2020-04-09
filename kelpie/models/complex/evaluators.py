import html
from kelpie.models.complex.complex import ComplEx
import numpy as np

class ComplExEvaluator:

    def eval(
            self,
            model: ComplEx,
            samples: np.array,
            write_output = False
    ):

        dataset = model.dataset

        # run prediction on the samples
        sample_2_scores, sample_2_ranks, sample_2_predictions = model.predict_samples(samples)

        all_ranks = []
        for sample in sample_2_ranks:
            head_rank, tail_rank = sample_2_ranks[sample]
            all_ranks.append(head_rank)
            all_ranks.append(tail_rank)

        mrr = self.mrr(all_ranks)
        h1 = self.hits_at(all_ranks, 1)

        if write_output:
            result_lines = []
            detail_lines = []
            for sample in sample_2_predictions:

                head_rank, tail_rank = sample_2_ranks[sample]
                head_prediction_ids, tail_prediction_ids = sample_2_predictions[sample]

                head_id, rel_id, tail_id = sample[0], sample[1], sample[2]
                head_name = dataset.get_name_for_entity_id()[head_id]
                rel_name = dataset.get_name_for_relation_id()[rel_id]
                tail_name = dataset.get_name_for_entity_id[tail_id]

                textual_fact_key = ";".join([head_name, rel_name, tail_name])

                result_lines.append(textual_fact_key + ";" + str(head_rank) + ";" + str(tail_rank) + "\n")

                head_prediction_names = [dataset.get_name_for_entity_id()[x] for x in head_prediction_ids]
                tail_prediction_names = [dataset.get_name_for_entity_id[x] for x in tail_prediction_ids]

                detail_lines.append(textual_fact_key + ";predict head;[" + ";".join(head_prediction_names) + "]\n")
                detail_lines.append(textual_fact_key + ";predict tail;[" + ";".join(tail_prediction_names) + "]\n")

            for i in range(len(result_lines)):
                result_lines[i] = html.unescape(result_lines[i])
            for i in range(len(detail_lines)):
                detail_lines[i] = html.unescape(detail_lines[i])

            with open("results.txt", "w") as output_file:
                output_file.writelines(result_lines)
            with open("details.txt", "w") as output_file:
                output_file.writelines(detail_lines)

        return mrr, h1

    def mrr(self, values):
        mrr = 0.0
        for value in values:
            mrr += 1.0 / float(value)
        mrr = mrr / float(len(values))
        return mrr

    def hits_at(self, values, k:int):
        hits = 0
        for value in values:
            if value <= k:
                hits += 1
        return float(hits) / float(len(values))

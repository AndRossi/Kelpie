import sys
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

parser.add_argument("--mode",
                    type=str,
                    choices=['necessary', 'sufficient'],
                    help="The mode for which to plot the explanation lengths: necessary or sufficient")

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the table or save it in Kelpie/reproducibility_images")

args = parser.parse_args()

def read_necessary_output_end_to_end(filepath):
    fact_to_explain_2_details = {}
    with open(filepath, "r") as input_file:
        input_lines = input_file.readlines()
        for line in input_lines:
            bits = line.strip().split(";")

            _head_to_explain, _rel_to_explain, _tail_to_explain = bits[0:3]
            _fact_to_explain = (_head_to_explain, _rel_to_explain, _tail_to_explain)
            _explanation_bits = bits[3:-4]
            assert len(_explanation_bits) % 3 == 0

            _explanation_facts = []
            i = 0
            while i < len(_explanation_bits):

                if _explanation_bits[i] != "":
                    _cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail = _explanation_bits[i], \
                                                                                   _explanation_bits[i + 1], \
                                                                                   _explanation_bits[i + 2]
                    _cur_expl_fact = (_cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail)
                    _explanation_facts.append(_cur_expl_fact)
                i += 3

            _explanation_facts = tuple(_explanation_facts)
            _original_score, _new_score = float(bits[-4]), float(bits[-3])
            _original_tail_rank, _new_tail_rank = float(bits[-2]), float(bits[-1])

            fact_to_explain_2_details[_fact_to_explain] = (
                _explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank)

    return fact_to_explain_2_details


def read_sufficient_output_end_to_end(filepath):
    fact_to_convert_2_details = {}
    fact_to_convert_2_original_fact_to_explain = {}
    with open(filepath, "r") as input_file:
        input_lines = input_file.readlines()
        for line in input_lines:
            bits = line.strip().split(";")
            _head_to_explain, _rel_to_explain, _tail_to_explain = bits[0:3]
            _head_to_convert, _rel_to_convert, _tail_to_convert = bits[3:6]

            _fact_to_explain = (_head_to_explain, _rel_to_explain, _tail_to_explain)
            _fact_to_convert = (_head_to_convert, _rel_to_convert, _tail_to_convert)

            _explanation_bits = bits[6:-4]
            assert len(_explanation_bits) % 3 == 0

            _explanation_facts = []
            i = 0
            while i < len(_explanation_bits):

                if _explanation_bits[i] != "":
                    _cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail = _explanation_bits[i], \
                                                                                   _explanation_bits[i + 1], \
                                                                                   _explanation_bits[i + 2]
                    _cur_expl_fact = (_cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail)
                    _explanation_facts.append(_cur_expl_fact)
                i += 3

            _explanation_facts = tuple(_explanation_facts)
            _original_score, _new_score = float(bits[-4]), float(bits[-3])
            _original_tail_rank, _new_tail_rank = float(bits[-2]), float(bits[-1])

            fact_to_convert_2_details[_fact_to_convert] = (
                _explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank)
            fact_to_convert_2_original_fact_to_explain[_fact_to_convert] = _fact_to_explain
    return fact_to_convert_2_details, fact_to_convert_2_original_fact_to_explain


def hits_at_k(ranks, k):
    count = 0.0
    for rank in ranks:
        if rank <= k:
            count += 1.0
    return round(count / float(len(ranks)), 3)


def mrr(ranks):
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1.0 / float(rank)
    return round(reciprocal_rank_sum / float(len(ranks)), 3)


def mr(ranks):
    rank_sum = 0.0
    for rank in ranks:
        rank_sum += float(rank)
    return round(rank_sum / float(len(ranks)), 3)


KELPIE_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))
END_TO_END_EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))

systems = ["K1", "Kelpie", "DP", "Criage"]
models = ["TransE", "ComplEx", "ConvE"]
datasets = ["FB15k", "WN18", "FB15k237", "WN18RR", "YAGO3-10"]
mode = args.mode
save = args.save

output_data = []
row_labels = []
for model in models:
    for system in systems:
        if model == "TransE" and system == "Criage":
            continue
        row_labels.append(f'{model} {system}')
        new_data_row = []
        for dataset in datasets:
            end_to_end_output_filename = "_".join(
                [system.lower(), mode.lower(), model.lower(), dataset.lower().replace("-", "")]) + ".csv"
            end_to_end_output_filepath = os.path.join(END_TO_END_EXPERIMENT_ROOT, end_to_end_output_filename)
            if not os.path.isfile(end_to_end_output_filepath):
                new_data_row.append("-")
                new_data_row.append("-")
                continue

            original_ranks = []
            new_ranks = []

            if mode == 'necessary':
                fact_2_kelpie_explanations = read_necessary_output_end_to_end(end_to_end_output_filepath)

                for fact_to_explain in fact_2_kelpie_explanations:
                    kelpie_expl, _, _, kelpie_original_tail_rank, kelpie_new_tail_rank = fact_2_kelpie_explanations[
                        fact_to_explain]

                    original_ranks.append(kelpie_original_tail_rank)
                    new_ranks.append(kelpie_new_tail_rank)

            else:
                fact_2_kelpie_explanations, _ = read_sufficient_output_end_to_end(end_to_end_output_filepath)

                for fact_to_convert in fact_2_kelpie_explanations:
                    kelpie_expl, _, _, kelpie_original_tail_rank, kelpie_new_tail_rank = fact_2_kelpie_explanations[
                        fact_to_convert]

                    original_ranks.append(kelpie_original_tail_rank)
                    new_ranks.append(kelpie_new_tail_rank)

            original_mrr, original_h1 = mrr(original_ranks), hits_at_k(original_ranks, 1)
            kelpie_mrr, kelpie_h1 = mrr(new_ranks), hits_at_k(new_ranks, 1)
            mrr_difference, h1_difference = round(kelpie_mrr - original_mrr, 3), round(kelpie_h1 - original_h1, 3)

            mrr_difference_str = "+" + str(mrr_difference) if mrr_difference > 0 else str(mrr_difference)
            h1_difference_str = "+" + str(h1_difference) if h1_difference > 0 else str(h1_difference)

            new_data_row.append(h1_difference_str)
            new_data_row.append(mrr_difference_str)
        output_data.append(new_data_row)

column_labels = ["Fb15k\nH@1", "FB15k\nMRR",
                 "WN18\nH@1", "WN18\nMRR",
                 "Fb15k237\nH@1", "FB15k237\nMRR",
                 "WN18RR\nH@1", "WN18RR\nMRR",
                 "YAGO3-10\nH@1", "YAGO3-10\nMRR"]

fig = plt.figure(figsize=(9, 3))
ax = fig.gca()
ax.axis('off')
table = ax.table(cellText=output_data,
                 loc="center",
                 rowLoc='center',
                 cellLoc='center',
                 colLabels=column_labels,
                 rowLabels=row_labels)

if not save:
    plt.show()
else:
    table.scale(1, 1.7)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    output_reproducibility_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_reproducibility_folder):
        os.makedirs(output_reproducibility_folder)
    output_path = os.path.join(output_reproducibility_folder, f'end_to_end_table_{args.mode}.png')
    print(f'Saving {args.mode} end-to-end results in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')

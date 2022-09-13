import sys
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the table or save it in Kelpie/reproducibility_images")

args = parser.parse_args()


def read_necessary_output_end_to_end(filepath):
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
        exit()

    fact_to_explain_2_details = {}
    with open(filepath, "r") as input_file:
        input_lines = input_file.readlines()
        for line in input_lines:
            bits = line.strip().split(";")

            _head_to_explain, _rel_to_explain, _tail_to_explain = bits[0:3]
            _fact_to_explain = (_head_to_explain, _rel_to_explain, _tail_to_explain)
            _explanation_bits = bits[3:-4]
            assert len(_explanation_bits)%3 == 0

            _explanation_facts = []
            i = 0
            while i < len(_explanation_bits):

                if _explanation_bits[i] != "":
                    _cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail = _explanation_bits[i], \
                                                                                _explanation_bits[i+1], \
                                                                                _explanation_bits[i+2]
                    _cur_expl_fact = (_cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail)
                    _explanation_facts.append(_cur_expl_fact)
                i+=3

            _explanation_facts = tuple(_explanation_facts)
            _original_score, _new_score = float(bits[-4]), float(bits[-3])
            _original_tail_rank, _new_tail_rank = float(bits[-2]), float(bits[-1])

            fact_to_explain_2_details[_fact_to_explain] = (_explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank)

    return fact_to_explain_2_details


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
END_TO_END_EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, "end_to_end"))
EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))

models = ["TransE", "ComplEx", "ConvE"]
datasets = ["FB15k", "WN18", "FB15k237", "WN18RR", "YAGO3-10"]
save = args.save

output_data = []
row_labels = []
for model in models:
    row_labels.append(f'{model}: ξ=1')
    row_labels.append(f'{model}: ξ=5')
    row_labels.append(f'{model}: ξ=10')

    xsi1_new_data_row = []
    xsi5_new_data_row = []
    xsi10_new_data_row = []

    for dataset in datasets:

        xsi1_filename = "_".join(['kelpie', model.lower(), dataset.lower().replace("-", ""), "1"]) + ".csv"
        xsi5_filename = "_".join(['kelpie', "necessary", model.lower(), dataset.lower().replace("-", "")]) + ".csv"
        xsi10_filename = "_".join(['kelpie', model.lower(), dataset.lower().replace("-", ""), "10"]) + ".csv"

        xsi1_filepath = os.path.join(EXPERIMENT_ROOT, xsi1_filename)
        xsi5_filepath = os.path.join(END_TO_END_EXPERIMENT_ROOT, xsi5_filename)
        xsi10_filepath = os.path.join(EXPERIMENT_ROOT, xsi10_filename)

        original_ranks = []
        xsi1_new_ranks = []
        xsi5_new_ranks = []
        xsi10_new_ranks = []

        fact_2_xsi1_explanations = read_necessary_output_end_to_end(xsi1_filepath)
        fact_2_xsi5_explanations = read_necessary_output_end_to_end(xsi5_filepath)
        fact_2_xsi10_explanations = read_necessary_output_end_to_end(xsi10_filepath)

        for fact_to_explain in fact_2_xsi1_explanations:
            xsi1_expl, _, _, original_tail_rank, xsi1_new_tail_rank = fact_2_xsi1_explanations[fact_to_explain]
            xsi5_expl, _, _, _, xsi5_new_tail_rank = fact_2_xsi5_explanations[fact_to_explain]
            xsi10_expl, _, _, _, xsi10_new_tail_rank = fact_2_xsi10_explanations[fact_to_explain]

            original_ranks.append(original_tail_rank)
            xsi1_new_ranks.append(xsi1_new_tail_rank)
            xsi5_new_ranks.append(xsi5_new_tail_rank)
            xsi10_new_ranks.append(xsi10_new_tail_rank)

        original_mrr = mrr(original_ranks)
        original_h1 = hits_at_k(original_ranks, 1)

        xsi1_mrr, xsi5_mrr, xsi10_mrr = mrr(xsi1_new_ranks), mrr(xsi5_new_ranks), mrr(xsi10_new_ranks)
        xsi1_h1, xsi5_h1, xsi10_h1 = hits_at_k(xsi1_new_ranks, 1), hits_at_k(xsi5_new_ranks, 1), hits_at_k(
            xsi10_new_ranks, 1)

        xsi1_mrr_diff = round(xsi1_mrr - original_mrr, 3)
        xsi1_h1_diff = round(xsi1_h1 - original_h1, 3)
        xsi1_mrr_diff_str = "+" + str(xsi1_mrr_diff) if xsi1_mrr_diff > 0 else str(xsi1_mrr_diff)
        xsi1_h1_diff_str = "+" + str(xsi1_h1_diff) if xsi1_h1_diff > 0 else str(xsi1_h1_diff)

        xsi5_mrr_diff = round(xsi5_mrr - original_mrr, 3)
        xsi5_h1_diff = round(xsi5_h1 - original_h1, 3)
        xsi5_mrr_diff_str = "+" + str(xsi5_mrr_diff) if xsi5_mrr_diff > 0 else str(xsi5_mrr_diff)
        xsi5_h1_diff_str = "+" + str(xsi5_h1_diff) if xsi5_h1_diff > 0 else str(xsi5_h1_diff)

        xsi10_mrr_diff = round(xsi10_mrr - original_mrr, 3)
        xsi10_h1_diff = round(xsi10_h1 - original_h1, 3)
        xsi10_mrr_diff_str = "+" + str(xsi10_mrr_diff) if xsi10_mrr_diff > 0 else str(xsi10_mrr_diff)
        xsi10_h1_diff_str = "+" + str(xsi10_h1_diff) if xsi10_h1_diff > 0 else str(xsi10_h1_diff)

        xsi1_new_data_row.append(xsi1_h1_diff_str)
        xsi1_new_data_row.append(xsi1_mrr_diff_str)
        xsi5_new_data_row.append(xsi5_h1_diff_str)
        xsi5_new_data_row.append(xsi5_mrr_diff_str)
        xsi10_new_data_row.append(xsi10_h1_diff_str)
        xsi10_new_data_row.append(xsi10_mrr_diff_str)

    output_data.append(xsi1_new_data_row)
    output_data.append(xsi5_new_data_row)
    output_data.append(xsi10_new_data_row)

column_labels = ["Fb15k\nΔH@1", "FB15k\nΔMRR",
                 "WN18\nΔH@1", "WN18\nΔMRR",
                 "Fb15k237\nΔH@1", "FB15k237\nΔMRR",
                 "WN18RR\nΔH@1", "WN18RR\nΔMRR",
                 "YAGO3-10\nΔH@1", "YAGO3-10\nΔMRR"]

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
    output_path = os.path.join(output_reproducibility_folder, f'xsi_threshold_comparison_table.png')
    print(f'Saving the comparison of different ξ threshold values in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')

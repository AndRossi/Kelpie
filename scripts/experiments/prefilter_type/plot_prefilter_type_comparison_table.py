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

parser.add_argument("--mode",
                    type=str,
                    choices=['necessary', 'sufficient'],
                    help="")

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


def read_sufficient_output_end_to_end(filepath):
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
        exit()

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

            fact_to_convert_2_details[_fact_to_convert] = (_explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank)
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
END_TO_END_EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, "end_to_end"))
PREFILTER_EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))

datasets = ["FB15k", "WN18", "FB15k237", "WN18RR", "YAGO3-10"]
save = args.save
mode = args.mode

output_data = []
row_labels = []
for model in ['ComplEx']:
    row_labels.append(f'{model}: topology-based')
    row_labels.append(f'{model}: type-based')

    topology_new_data_row = []
    typebased_new_data_row = []

    for dataset in datasets:

        topology_filename = "_".join(['kelpie', mode.lower(), model.lower(), dataset.lower().replace("-", "")]) + ".csv"
        typebased_filename = "_".join(['kelpie', mode.lower(), model.lower(), dataset.lower().replace("-", ""), "typebased_prefilter"]) + ".csv"

        topology_filepath = os.path.join(END_TO_END_EXPERIMENT_ROOT, topology_filename)
        typebased_filepath = os.path.join(PREFILTER_EXPERIMENT_ROOT, typebased_filename)

        original_ranks = []
        topology_new_ranks = []
        typebased_new_ranks = []

        fact_2_topology_explanations = read_necessary_output_end_to_end(topology_filepath)
        fact_2_typebased_explanations = read_necessary_output_end_to_end(typebased_filepath)

        for fact_to_explain in fact_2_topology_explanations:
            topology_expl, _, _, original_tail_rank, topology_new_tail_rank = fact_2_topology_explanations[fact_to_explain]
            typebased_expl, _, _, _, typebased_new_tail_rank = fact_2_typebased_explanations[fact_to_explain]

            original_ranks.append(original_tail_rank)
            topology_new_ranks.append(topology_new_tail_rank)
            typebased_new_ranks.append(typebased_new_tail_rank)

        original_mrr = mrr(original_ranks)
        original_h1 = hits_at_k(original_ranks, 1)

        topology_mrr, typebased_mrr = mrr(topology_new_ranks), mrr(typebased_new_ranks)
        topology_h1, typebased_h1 = hits_at_k(topology_new_ranks, 1), hits_at_k(typebased_new_ranks, 1)

        topology_mrr_diff = round(topology_mrr - original_mrr, 3)
        topology_h1_diff = round(topology_h1 - original_h1, 3)
        topology_mrr_diff_str = "+" + str(topology_mrr_diff) if topology_mrr_diff > 0 else str(topology_mrr_diff)
        topology_h1_diff_str = "+" + str(topology_h1_diff) if topology_h1_diff > 0 else str(topology_h1_diff)

        typebased_mrr_diff = round(typebased_mrr - original_mrr, 3)
        typebased_h1_diff = round(typebased_h1 - original_h1, 3)
        typebased_mrr_diff_str = "+" + str(typebased_mrr_diff) if typebased_mrr_diff > 0 else str(typebased_mrr_diff)
        typebased_h1_diff_str = "+" + str(typebased_h1_diff) if typebased_h1_diff > 0 else str(typebased_h1_diff)

        topology_new_data_row.append(topology_h1_diff_str)
        topology_new_data_row.append(topology_mrr_diff_str)
        typebased_new_data_row.append(typebased_h1_diff_str)
        typebased_new_data_row.append(typebased_mrr_diff_str)

    output_data.append(topology_new_data_row)
    output_data.append(typebased_new_data_row)

column_labels = ["Fb15k\nΔH@1", "FB15k\nΔMRR",
                 "WN18\nΔH@1", "WN18\nΔMRR",
                 "Fb15k237\nΔH@1", "FB15k237\nΔMRR",
                 "WN18RR\nΔH@1", "WN18RR\nΔMRR",
                 "YAGO3-10\nΔH@1", "YAGO3-10\nΔMRR"]

fig = plt.figure(figsize=(9, 1))
ax = fig.gca()
ax.axis('off')
table = ax.table(cellText=output_data,
                 loc="center",
                 rowLoc='center',
                 cellLoc='center',
                 colLabels=column_labels,
                 rowLabels=row_labels)

if not save:
    table.scale(1, 1.7)
    plt.subplots_adjust(left=0.24, bottom=0.15)
    plt.show()
else:
    table.scale(1, 1.7)
    plt.subplots_adjust(left=0.24, bottom=0.15)
    output_reproducibility_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_reproducibility_folder):
        os.makedirs(output_reproducibility_folder)
    output_path = os.path.join(output_reproducibility_folder, f'prefilter_type_comparison_table_{mode}.png')
    print(f'Saving {mode} comparison of different prefilter types in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')

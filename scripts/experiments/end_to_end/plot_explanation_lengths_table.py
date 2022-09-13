import os
import sys
import argparse
import numpy
import matplotlib.pyplot as plt

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))


from dataset import ALL_DATASET_NAMES

COMPLEX = "ComplEx"
CONVE = "ConvE"
TRANSE = "TransE"

COLOR_1 = "#2196f3"
COLOR_2 = "#8bc34a"
COLOR_3 = "#ffc107"
COLOR_4 = "#f44336"


def read_necessary_output_end_to_end(filepath):
    fact_to_explain_2_explanation_length = {}
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
            _explanation_length = len(_explanation_facts)
            fact_to_explain_2_explanation_length[_fact_to_explain] = _explanation_length

    return fact_to_explain_2_explanation_length


def read_sufficient_output_end_to_end(filepath):
    fact_to_convert_2_explanation_length = {}
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

            _explanation_length = len(_explanation_facts)
            fact_to_convert_2_explanation_length[_fact_to_convert] = _explanation_length
    return fact_to_convert_2_explanation_length


KELPIE_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))
END_TO_END_EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))


def extract_lengths(lengths):
    output = {1:0, 2:0, 3:0, 4:0}
    for length in lengths:
        output[length] += 1
    return output

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument("--mode",
                    type=str,
                    choices=["necessary", "sufficient"],
                    required=True,
                    help="The mode for which to plot the explanation lengths: necessary or sufficient")

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the table or save it in Kelpie/reproducibility_images")


models = ["TransE", "ComplEx", "ConvE"]
modes = ["necessary", "sufficient"]
datasets = ["FB15k", "WN18", "FB15k-237", "WN18RR", "YAGO3-10"]

counts = {}
for _model in models:
    counts[_model] = {}
    for _mode in modes:
        counts[_model][_mode] = {}
        for _dataset in datasets:
            end_to_end_output_filename = "_".join(
                    ["kelpie", _mode.lower(), _model.lower(), _dataset.lower().replace("-", "")]) + ".csv"

            end_to_end_output_filepath = os.path.join(END_TO_END_EXPERIMENT_ROOT, end_to_end_output_filename)
            if _mode == "necessary":
                fact_2_explanation_lengths = read_necessary_output_end_to_end(end_to_end_output_filepath)
                counts[_model][_mode][_dataset] = list(fact_2_explanation_lengths.values())
            else:
                fact_2_explanation_lengths = read_sufficient_output_end_to_end(end_to_end_output_filepath)
                counts[_model][_mode][_dataset] = list(fact_2_explanation_lengths.values())

args = parser.parse_args()
mode = args.mode

output_rows = []
for model in models:
    new_row = []
    for dataset in datasets:
        cur_lengths = counts[model][args.mode][dataset]
        new_row.append(round(numpy.average(cur_lengths), 2))
        new_row.append(round(numpy.std(cur_lengths), 2))
    output_rows.append(new_row)

column_labels = ["Fb15k\nAVG", "Fb15k\nSTD",
                 "WN18\nAVG", "WN18\nSTD",
                 "FB15k-237\nAVG", "FB15k-237\nSTD",
                 "WN18RR\nAVG", "WN18RR\nSTD",
                 "YAGO3-10\nAVG", "YAGO3-10\nSTD"]

row_labels = ['TransE', 'ComplEx', 'ConvE']
fig = plt.figure(figsize=(9, 1.5))
ax = fig.gca()
ax.axis('off')
table = ax.table(cellText=output_rows,
                 loc="center",
                 rowLoc='center',
                 cellLoc='center',
                 colLabels=column_labels,
                 rowLabels=row_labels)

if not args.save:
    plt.show()
else:
    table.scale(1, 1.7)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    output_reproducibility_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_reproducibility_folder):
        os.makedirs(output_reproducibility_folder)
    output_path = os.path.join(output_reproducibility_folder, f'explanation_lengths_table_{args.mode}.png')
    print(f'Saving {args.mode} explanation lengths in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')

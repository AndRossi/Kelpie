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
    fact_to_explain_2_facts_to_convert = {}
    with open(filepath, "r") as input_file:
        input_lines = input_file.readlines()
        for line in input_lines:
            bits = line.strip().split(";")
            _head_to_explain, _rel_to_explain, _tail_to_explain = bits[0:3]
            _head_to_convert, _rel_to_convert, _tail_to_convert = bits[3:6]

            _fact_to_explain = (_head_to_explain, _rel_to_explain, _tail_to_explain)
            _fact_to_convert = (_head_to_convert, _rel_to_convert, _tail_to_convert)

            if _fact_to_explain not in fact_to_explain_2_facts_to_convert:
                fact_to_explain_2_facts_to_convert[_fact_to_explain] = []
            fact_to_explain_2_facts_to_convert[_fact_to_explain].append(_fact_to_convert)

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

    fact_to_explain_2_explanation_length = {}
    for _fact_to_explain in fact_to_explain_2_facts_to_convert:
        _fact_to_convert = fact_to_explain_2_facts_to_convert[_fact_to_explain][0]
        expl_length = fact_to_convert_2_explanation_length[_fact_to_convert]
        fact_to_explain_2_explanation_length[_fact_to_explain] = expl_length

    return fact_to_explain_2_explanation_length


KELPIE_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))
END_TO_END_EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))


def aggregate_lengths(lengths):
    output = {1:0, 2:0, 3:0, 4:0}
    for length in lengths:
        output[length] += 1
    return output


parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

ALL_MODEL_NAMES = ["ComplEx", "ConvE", "TransE"]
parser.add_argument("--model",
                    type=str,
                    choices=ALL_MODEL_NAMES,
                    required=True,
                    help="The model for which to plot the explanation lengths:: ComplEx, ConvE, or TransE")

parser.add_argument("--mode",
                    type=str,
                    choices=["necessary", "sufficient"],
                    required=True,
                    help="The mode for which to plot the explanation lengths: necessary or sufficient")

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the plot or save it in Kelpie/reproducibility_images")

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
                counts[_model][_mode][_dataset] = aggregate_lengths(fact_2_explanation_lengths.values())
            else:
                fact_2_explanation_lengths = read_sufficient_output_end_to_end(end_to_end_output_filepath)
                counts[_model][_mode][_dataset] = aggregate_lengths(fact_2_explanation_lengths.values())

args = parser.parse_args()
mode = args.mode
model = args.model
save = args.save

counts_to_show = counts[model][mode]

labels = ALL_DATASET_NAMES
counts_to_show_1 = [counts_to_show[dataset_name][1] for dataset_name in labels]
counts_to_show_2 = [counts_to_show[dataset_name][2] for dataset_name in labels]
counts_to_show_3 = [counts_to_show[dataset_name][3] for dataset_name in labels]
counts_to_show_4 = [counts_to_show[dataset_name][4] for dataset_name in labels]
x = numpy.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5 * width, counts_to_show_1, width, label='1', zorder=3)
rects2 = ax.bar(x - 0.5 * width, counts_to_show_2, width, label='2', zorder=3)
rects3 = ax.bar(x + 0.5 * width, counts_to_show_3, width, label='3', zorder=3)
rects4 = ax.bar(x + 1.5 * width, counts_to_show_4, width, label='4', zorder=3)

for i in range(len(rects1)):
    rects1[i].set_color(COLOR_1)
    rects2[i].set_color(COLOR_2)
    rects3[i].set_color(COLOR_3)
    rects4[i].set_color(COLOR_4)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

ax.set_ylabel('Number of explanations grouped by size', fontsize=10)
ax.set_xticks(x)
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.ylim([0, 100])
ax.set_xticklabels(labels, rotation='horizontal')
ax.grid(zorder=0, axis="y")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)

ax.set_title("Explanation lengths for the " + str(model) + " model in " + str(mode) + " scenario", pad=20)

fig.tight_layout()

if not save:
    plt.show()
else:
    output_reproducibility_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_reproducibility_folder):
        os.makedirs(output_reproducibility_folder)
    output_path = os.path.join(output_reproducibility_folder, f'explanation_lengths_plot_{model.lower()}_{mode.lower()}.png')
    print(f'Saving {args.mode} explanation lengths plot in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')

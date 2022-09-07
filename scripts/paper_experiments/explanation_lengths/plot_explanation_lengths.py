import os
import sys
import argparse
import numpy
import matplotlib.pyplot as plt

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from dataset import FB15K, WN18, FB15K_237, WN18RR, YAGO3_10, ALL_DATASET_NAMES

COMPLEX = "ComplEx"
CONVE = "ConvE"
TRANSE = "TransE"

COLOR_1 = "#2196f3"
COLOR_2 = "#8bc34a"
COLOR_3 = "#ffc107"
COLOR_4 = "#f44336"

counts = {COMPLEX: {"necessary": {FB15K: {'1': 15, '2': 4, '3': 7, '4': 74},
                                  FB15K_237: {'1': 2, '2': 3, '3': 5, '4': 90},
                                  WN18: {'1': 12, '2': 2, '3': 21, '4': 65},
                                  WN18RR: {'1': 12, '2': 16, '3': 16, '4': 56},
                                  YAGO3_10: {'1': 45, '2': 14, '3': 10, '4': 31}},
                    'sufficient': {FB15K: {'1': 72, '2': 20, '3': 4, '4': 4},
                                   FB15K_237: {'1': 26, '2': 30, '3': 11, '4': 33},
                                   WN18: {'1': 100, '2': 0, '3': 0, '4': 0},
                                   WN18RR: {'1': 100, '2': 0, '3': 0, '4': 0},
                                   YAGO3_10: {'1': 98, '2': 1, '3': 0, '4': 1}}},
          CONVE: {"necessary": {FB15K: {'1': 5, '2': 13, '3': 23, '4': 59},
                                FB15K_237: {'1': 44, '2': 9, '3': 24, '4': 23},
                                WN18: {'1': 22, '2': 11, '3': 21, '4': 46},
                                WN18RR: {'1': 28, '2': 12, '3': 26, '4': 34},
                                YAGO3_10: {'1': 56, '2': 25, '3': 10, '4': 9}},
                  'sufficient': {FB15K: {'1': 64, '2': 9, '3': 7, '4': 20},
                                 FB15K_237: {'1': 13, '2': 17, '3': 18, '4': 52},
                                 WN18: {'1': 99, '2': 1, '3': 0, '4': 0},
                                 WN18RR: {'1': 97, '2': 2, '3': 1, '4': 0},
                                 YAGO3_10: {'1': 86, '2': 10, '3': 4, '4': 0}}},
          TRANSE: {"necessary": {FB15K: {'1': 20, '2': 17, '3': 28, '4': 35},
                                 FB15K_237: {'1': 27, '2': 23, '3': 23, '4': 27},
                                 WN18: {'1': 48, '2': 9, '3': 21, '4': 22},
                                 WN18RR: {'1': 58, '2': 27, '3': 5, '4': 10},
                                 YAGO3_10: {'1': 53, '2': 14, '3': 14, '4': 19}},
                   'sufficient': {FB15K: {'1': 46, '2': 32, '3': 9, '4': 13},
                                  FB15K_237: {'1': 2, '2': 16, '3': 19, '4': 63},
                                  WN18: {'1': 98, '2': 2, '3': 0, '4': 0},
                                  WN18RR: {'1': 48, '2': 42, '3': 5, '4': 5},
                                  YAGO3_10: {'1': 66, '2': 26, '3': 5, '4': 3}}}}

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

args = parser.parse_args()
model = args.model
mode = args.mode


counts_to_show = counts[model][mode]
labels = ALL_DATASET_NAMES
counts_to_show_1 = [counts_to_show[dataset_name]['1'] for dataset_name in labels]
counts_to_show_2 = [counts_to_show[dataset_name]['2'] for dataset_name in labels]
counts_to_show_3 = [counts_to_show[dataset_name]['3'] for dataset_name in labels]
counts_to_show_4 = [counts_to_show[dataset_name]['4'] for dataset_name in labels]
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

plt.show()

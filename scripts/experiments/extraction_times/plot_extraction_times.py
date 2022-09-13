import argparse
import sys
import os
import numpy
import matplotlib.pyplot as plt

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from dataset import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10

KELPIE_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))

COMPLEX = "ComplEx"
CONVE = "ConvE"
TRANSE = "TransE"

labels = [COMPLEX, CONVE, TRANSE]

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument("--mode",
                    type=str,
                    choices=['necessary', 'sufficient'],
                    help="The mode of the explanations to plot the extraction times of, either necessary or sufficient")

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the plot or save it in Kelpie/reproducibility_images")

args = parser.parse_args()

mode = args.mode
save = args.save

necessary_execution_times = {COMPLEX: {FB15K: 1222, WN18: 27, FB15K_237: 145, WN18RR: 39, YAGO3_10: 133},
                             CONVE: {FB15K: 2019, WN18: 127, FB15K_237: 34, WN18RR: 28, YAGO3_10: 128},
                             TRANSE: {FB15K: 78, WN18: 50, FB15K_237: 24, WN18RR: 28, YAGO3_10: 128} }

sufficient_execution_times = {COMPLEX: {FB15K: 1625, WN18: 86, FB15K_237: 687, WN18RR: 112, YAGO3_10: 763},
                             CONVE: {FB15K: 5357, WN18: 268, FB15K_237: 328, WN18RR: 136, YAGO3_10: 1097},
                             TRANSE: {FB15K: 437, WN18: 155, FB15K_237: 437, WN18RR: 69, YAGO3_10: 551} }

if args.mode == 'sufficient':
    execution_times_fb15k = [sufficient_execution_times[model_name][FB15K] for model_name in labels]
    execution_times_fb15k237 = [sufficient_execution_times[model_name][FB15K_237] for model_name in labels]
    execution_times_wn18 = [sufficient_execution_times[model_name][WN18] for model_name in labels]
    execution_times_wn18rr = [sufficient_execution_times[model_name][WN18RR] for model_name in labels]
    execution_times_yago310 = [sufficient_execution_times[model_name][YAGO3_10] for model_name in labels]
else:
    execution_times_fb15k = [necessary_execution_times[model_name][FB15K] for model_name in labels]
    execution_times_fb15k237 = [necessary_execution_times[model_name][FB15K_237] for model_name in labels]
    execution_times_wn18 = [necessary_execution_times[model_name][WN18] for model_name in labels]
    execution_times_wn18rr = [necessary_execution_times[model_name][WN18RR] for model_name in labels]
    execution_times_yago310 = [necessary_execution_times[model_name][YAGO3_10] for model_name in labels]

x = numpy.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 2*width, execution_times_fb15k, width, label=FB15K, zorder=3)
rects2 = ax.bar(x - 1*width, execution_times_fb15k237, width, label=FB15K_237, zorder=3)
rects3 = ax.bar(x, execution_times_wn18, width, label=WN18, zorder=3)
rects4 = ax.bar(x + 1*width, execution_times_wn18rr, width, label=WN18RR, zorder=3)
rects5 = ax.bar(x + 2*width, execution_times_yago310, width, label=YAGO3_10, zorder=3)

FB15K_COLOR="#ED553B"
FB15K237_COLOR="#e8c441"
WN18_COLOR="#3CAEA3"
WN18RR_COLOR="#20639B"
YAGO310_COLOR="#173F5F"


for i in range(len(rects1)):
    rects1[i].set_color(FB15K_COLOR)
    rects2[i].set_color(FB15K237_COLOR)
    rects3[i].set_color(WN18_COLOR)
    rects4[i].set_color(WN18RR_COLOR)
    rects5[i].set_color(YAGO310_COLOR)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.tick_params(axis="x")
ax.tick_params(axis="y")

ax.set_title(args.mode.capitalize() + ' explanation extraction times', pad=20)

ax.set_ylabel('Extraction times in seconds', fontsize=10.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='horizontal', fontsize=10)
ax.grid(zorder=0)
ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=10)

ax.set_yscale('log')

fig.tight_layout()

if not save:
    plt.show()
if save:
    output_reproducibility_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_reproducibility_folder):
        os.makedirs(output_reproducibility_folder)
    output_path = os.path.join(output_reproducibility_folder, f'extraction_times_{mode}.png')
    print(f'Saving {args.mode} explanation extraction time plot in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')

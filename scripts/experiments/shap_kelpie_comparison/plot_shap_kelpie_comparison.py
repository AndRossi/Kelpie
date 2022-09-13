import numpy
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the plot or save it in Kelpie/reproducibility_images")

args = parser.parse_args()
save = args.save

SHAPLEY = "Shapley Values"
KERNELSHAP = "KernelSHAP"
KELPIE = "Our Explanation Builder"
labels = ["Fact 1", "Fact 2", "Fact 3", "Fact 4", "Fact 5", "Fact 6", "Fact 7", "Fact 8", "Fact 9", "Fact 10"]

input_fact_2_visits = {
    ("/m/06rny", "/american_football/football_team/current_roster./american_football/football_roster_position/position", "/m/023wyl"): (1048576, 566755, 108),
    ("/m/02q1tc5", "/award/award_category/winners./award/award_honor/award_winner", "/m/026dg51"): (1048576, 569366, 70),
    ("/m/02wkmx", "/award/award_category/winners./award/award_honor/award_winner", "/m/049l7"): (1048576, 551479, 20),
    ("/m/011ycb", "/award/award_nominated_work/award_nominations./award/award_nomination/award", "/m/04dn09n"): (1048576, 468673, 75),
    ("/m/04ddm4", "/award/award_nominated_work/award_nominations./award/award_nomination/award", "/m/05b1610"): (1048576, 414206, 170),
    ("/m/03np63f", "/award/award_nominated_work/award_nominations./award/award_nomination/award", "/m/02y_rq5"): (1048576, 394543, 20),
    ("/m/072192", "/award/award_nominated_work/award_nominations./award/award_nomination/award", "/m/0k611"): (1048576, 560200, 54),
    ("/m/011yl_", "/award/award_nominated_work/award_nominations./award/award_nomination/award", "/m/099c8n"): (1048576, 474501, 60),
    ("/m/0m313", "/award/award_nominated_work/award_nominations./award/award_nomination/award_nominee", "/m/0m31m"): (1048576, 305706, 58),
    ("/m/06yykb", "/award/award_nominated_work/award_nominations./award/award_nomination/award_nominee", "/m/01vvb4m"): (1048576, 596037, 73)}

shapley_value_visits = [input_fact_2_visits[x][0] for x in input_fact_2_visits]
kernelshap_visits = [input_fact_2_visits[x][1] for x in input_fact_2_visits]
kelpie_visits = [input_fact_2_visits[x][2] for x in input_fact_2_visits]

x = numpy.arange(len(shapley_value_visits))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(9, 5))

rects1 = ax.bar(x - width, shapley_value_visits, width, label=SHAPLEY, zorder=3)
rects2 = ax.bar(x, kernelshap_visits, width, label=KERNELSHAP, zorder=3)
rects3 = ax.bar(x + width, kelpie_visits, width, label=KELPIE, zorder=3)

SHAPLEY_COLOR="#ED553B"
KERNELSHAP_COLOR="#e8c441"
KELPIE_COLOR="#3CAEA3"
WN18RR_COLOR="#20639B"
YAGO310_COLOR="#173F5F"


for i in range(len(rects1)):
    rects1[i].set_color(SHAPLEY_COLOR)
    rects2[i].set_color(KERNELSHAP_COLOR)
    rects3[i].set_color(KELPIE_COLOR)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

ax.set_ylabel('Visited combinations of Pre-Filtered facts', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='horizontal')
ax.grid(zorder=0)
ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=10)

ax.set_yscale('log')
ax.set_title("Comparison among Shalpey Values, KernelSHAP and Kelpie.\n "
             "Extractions for 10 random ComplEx predictions on FB15k.", pad=15)

fig.tight_layout()


KELPIE_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))

if not save:
    plt.show()
else:
    output_reproducibility_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_reproducibility_folder):
        os.makedirs(output_reproducibility_folder)
    output_path = os.path.join(output_reproducibility_folder, f'shap_kelpie_comparison.png')
    print(f'Saving efficiency comparison of different explanation building poliies in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')





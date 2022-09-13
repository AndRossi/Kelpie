import matplotlib.pyplot as plt
import argparse
import os

KELPIE_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the plot or save it in Kelpie/reproducibility_images")

args = parser.parse_args()
save = args.save

prefilter_times = {5: 26.6946750164032,
                   10: 76.38848857879638,
                   25: 84.43329257965088,
                   50: 147.43361716270448,
                   100: 181.69561727046965,
                   150: 344.26280949115755,
                   200: 459.26806592941284,
                   250: 787.5852595090867,
                   300: 718.4076545953751,
                   350: 1350.3705957174302 }

no_prefilter_times = {5: 26.916694355010986,
                      10: 76.59263441562652,
                      25: 89.58962974548339,
                      50: 214.93507361412048,
                      100: 304.47244517803193,
                      150: 721.5794765233993,
                      200: 1504.0622381210328,
                      250: 3268.637953567505,
                      300: 7929.163806319237,
                      350: 19162.33853366375 }


prefilter_x = [5, 10, 25, 50, 100, 150, 200, 250, 300, 350]
prefilter_y = [prefilter_times[x] for x in prefilter_x]
no_prefilter_x = [5, 10, 25, 50, 100, 150, 200, 250, 300, 350]
no_prefilter_y = [no_prefilter_times[x] for x in no_prefilter_x]

plt.plot(no_prefilter_x, no_prefilter_y, color="#9D2EC5", linewidth=3, label="Not Using Pre-Filter")
plt.plot(prefilter_x, prefilter_y, color="#F5B14C", linewidth=3, label="Using Pre-Filter")

plt.grid()
plt.legend()

plt.ylabel('Necessary explanation extraction time')
plt.xlabel('Training mentions of the entity to post-train')

if save:
    output_reproducibility_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_reproducibility_folder):
        os.makedirs(output_reproducibility_folder)
    output_path = os.path.join(output_reproducibility_folder, f'extraction_times_with_and_without_prefilter_plot.png')
    print(f'Saving plot of explanation extraction times with and without prefilter in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')
else:
    plt.show()

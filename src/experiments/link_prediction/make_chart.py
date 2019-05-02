import matplotlib.pyplot as plt
import json
import os

# The script for creating charts out of link prediction results.
# Edit the [NAMING] schemes and uncomment one of the [CHARTS] sections
# to generate required chart

# <<< CONSTANTS >>>
DATASET = "Bugzilla"
MIN_K = 5
OUTDIR = "./output"
MARKERS = ["v", "*", "^", "X", "D"]

# <<< CHARTS  CONFIGURATIONS>>>
'''
XLABEL = "k"
YLABEL = "Precision@k [%]"
TITLE = f"Link Prediction - {DATASET}"
METRIC = "PRECISIONS"

'''
XLABEL = "k"
YLABEL = "Mean Squared Error @ k (MSE)"
TITLE = f"Link Prediction - {DATASET}"
METRIC = "MSE_OBS"



def get_result(file):
    # Remove .txt to get method name
    method_name = file[:-4]

    # Load json
    with open(f"{OUTDIR}/{file}") as res_file:
        return method_name, json.load(res_file)


def draw_results(marker, name, result):
    res = result[METRIC]
    rng = range(len(res))

    plt.plot(rng[MIN_K:], res[MIN_K:], linewidth=2, marker=marker, label=name)


if __name__ == "__main__":
    files = os.listdir(OUTDIR)

    for i, f in enumerate(files):
        r = get_result(f)
        draw_results(MARKERS[i], *r)

    plt.title(TITLE)
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Save plot
    outfile = f"lpr_{DATASET}_{METRIC}.png".lower()
    plt.savefig(outfile, bbox_inches="tight")

    # Draw plot
    plt.subplots_adjust(right=0.6, top=0.8, bottom=0.2)
    plt.show()

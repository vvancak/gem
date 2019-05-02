import matplotlib.pyplot as plt
import json
import os

# The script for creating charts out of graph reconstruction results.
# Edit the [NAMING] schemes and uncomment one of the [CHARTS] sections
# to generate required chart

# <<< CONSTANTS >>>
DATASET = "Bugzilla"
OUTDIR = "./output"
MARKERS = [".", "v", "*", "^", "X", "D"]

# <<< CHARTS CONFIGURATIONS >>>
XLABEL = "Graph Embedding Dimension"
YLABEL = "Mean Average Precision (MAP) [%]"
TITLE = f"Deep Walk - Adding Weights - {DATASET}"
METRIC = "MAP"


def get_result(file):
    # Remove .txt to get method name
    method_name = file[:-4]

    # Load json
    with open(f"{OUTDIR}/{file}") as res_file:
        return method_name, json.load(res_file)


def draw_results(marker, name, result):
    dims = []
    maps = []
    for dr in result:
        dims.append(dr["dimension"])
        maps.append(dr["results"][METRIC])

    plt.plot(dims, maps, linewidth=2, marker=marker, label=name)


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
    outfile = f"dwi_{DATASET}_{METRIC}.png".lower()
    plt.savefig(outfile, bbox_inches="tight")

    # Draw plot
    plt.subplots_adjust(right=0.6, top=0.8, bottom=0.2)
    plt.show()

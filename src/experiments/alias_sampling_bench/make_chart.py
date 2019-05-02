import experiments.alias_sampling_bench.evaluate as ev
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# The script for creating charts out of graph reconstruction results.
# Edit the [NAMING] schemes and uncomment one of the [CHARTS] sections
# to generate required chart

# <<< CONSTANTS >>>
OUTDIR = "./output"
MARKERS = ["X", "."]

# <<< CHARTS CONFIGURATIONS >>>
XLABEL = "Number of Classes (log scale)"
YLABEL = "Average time per 10 000 draws [ms]"
TITLE = f"Alias Sampling vs NumPy Random"


def get_result(file):
    # Remove .txt to get method name
    method_name = file[:-4]

    # Load json
    with open(f"{OUTDIR}/{file}") as res_file:
        return method_name, json.load(res_file)


def draw_results(marker, name, result):
    dims = []
    times = []
    for dr in result:
        dims.append(dr["classes"])
        times.append(dr["draw_time"])

    plt.plot(dims, np.array(times) * 1000, linewidth=2, marker=marker, label=name)


if __name__ == "__main__":
    files = os.listdir(OUTDIR)

    plt.figure(figsize=(6, 3), dpi=200)
    plt.subplot(111)

    plt.title(TITLE)
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.xscale('log')

    for i, f in enumerate(files):
        r = get_result(f)
        draw_results(MARKERS[i], *r)

    plt.xticks(ev.CLASSES, ev.CLASSES)
    plt.yticks([10, 50, 70, 100, 150, 200, 250, 300], [10, 50, 70, 100, 150, 200, 250, 300])
    plt.legend(loc="center")

    # Save plot
    outfile = f"als_bench.png".lower()
    plt.savefig(outfile, bbox_inches="tight")

    # Draw plot
    plt.show()

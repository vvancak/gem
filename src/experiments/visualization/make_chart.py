import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

# The script for creating charts out of graph reconstruction results.
# Edit the [NAMING] schemes and uncomment one of the [CHARTS] sections
# to generate required chart

# <<< CONSTANTS >>>
DATASET = "zachary_karate"
OUTDIR = "./output"

# <<< CHARTS CONFIGURATIONS >>>
XLABEL = "X"
YLABEL = "Y"

NROWS = 3
NCOLS = 2


def get_result(file):
    method_name = file[:-4]
    img = mpimg.imread(f"{OUTDIR}/{file}")
    return (method_name, img)


if __name__ == "__main__":
    files = os.listdir(OUTDIR)

    # gather results
    plots = []
    for i, f in enumerate(files):
        r = get_result(f)
        plots.append(r)

    #  plot
    fig = plt.figure(figsize=(NCOLS * 6, NROWS * 4.5), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0.2)
    for i, (name, plot) in enumerate(plots):
        a = fig.add_subplot(NROWS, NCOLS, i+1)
        imgplot = plt.imshow(plot)
        plt.axis('off')
        a.set_title(name)

    # Save & Show plot
    outfile = f"vis_{DATASET}.png".lower()

    plt.savefig(outfile, bbox_inches='tight')
    plt.show()

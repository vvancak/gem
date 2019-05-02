import sys

sys.path.insert(0, "../../")

import run.configurations as cfg
import run.print_headers as ph
import run.run_stages as rs
import typing as t
import json
import os

# === DATASET CONFIGUTATION ===
DATASET = "UW-bugzilla"
NORMALIZATION = None

DIMENSIONS = [2, 3, 4, 5]
SEEDS = [42, 24, 31, 9, 8, 96, 21, 19, 6, 23, 2, 10, 22, 14, 5, 84, 34, 17, 51, 91]

METHODS = ["SDNE_DBN", "SDNE"]


# === EVALUATION ===
def process_results(results: t.List) -> t.Dict:
    RES_FIELDS = ["MAP", "MSE_OBS", "time"]

    # Initialize to 0
    avg_res = {}
    for r in RES_FIELDS:
        avg_res[r] = 0.0

    # Sum everything
    for result in results:
        for r in RES_FIELDS:
            avg_res[r] += result[r]

    # Average
    for r in RES_FIELDS:
        avg_res[r] = round(avg_res[r] / REPETITIONS, 3)

    return avg_res


if __name__ == "__main__":
    config = cfg.Configuration(f"./config")

    # Configurations
    ds_config = config.ds_config()
    em_config = config.em_config(DATASET)
    ev_config = config.ev_config(DATASET)

    os.makedirs(f"./output/", exist_ok=True)

    # Runner
    runner = rs.RunStages(0, DATASET, None, None)
    graphs = runner.load_graphs(ds_config, NORMALIZATION)

    for m in METHODS:
        print(f" <<< === {m} === >>>")

        method_results = []
        for dim in DIMENSIONS:
            print(f"{ph.INFO} Evaluating for dimension {dim}")

            dim_results = []
            dim_times = []
            for s in SEEDS:
                # Learn embedding
                runner = rs.RunStages(s, DATASET, m, "graph_reconstruction")
                runner.set_graphs(*graphs)
                embedding, time = runner.learn_embedding(em_config, dim)

                # Evaluate embedding
                result = runner.evaluate_embedding(ev_config)
                dim_results.append({**result, "time": time})

            method_results.append({"dimension": dim, "results": process_results(dim_results)})

        # Save Results
        with open(f"./output/{m}.txt", "w+") as fp:
            json.dump(method_results, fp)

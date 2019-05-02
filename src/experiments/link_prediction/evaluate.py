import sys

sys.path.insert(0, "../../")

import run.configurations as cfg
import run.run_stages as rs
import typing as t
import numpy as np
import json
import os

# === DATASET CONFIGURATIONS ===
BUGZILLA_CONFIG = {
    "DATASET": "UW-bugzilla",
    "NORMALIZATION": None,
    "HIDE_PERC": 20,
    "DIMENSION": 4,
    "SEEDS": [42, 24, 31, 9, 8, 96, 21, 19, 6, 23, 2, 10, 22, 14, 5, 84, 34, 17, 51, 91],
    "METHODS": ["laplacian_eigenmaps", "deep_walk", "node2vec", "LINE", "SDNE"]
}

BANK_CONFIG = {
    "DATASET": "UW-bank_clients",
    "NORMALIZATION": None,
    "HIDE_PERC": 20,
    "DIMENSION": 128,
    "SEEDS": [42],
    "METHODS": ["deep_walk"]  # , "node2vec", "LINE", "SDNE"]
}

# === SELECT WHICH DATASET ===
USE_CONFIG = BANK_CONFIG


# === EVALUATION ===
def process_results(results: t.List, max_k: int) -> t.Dict:
    RES_FIELDS = ["PRECISIONS", "MSE_OBS"]

    rd = {}
    for f in RES_FIELDS:
        rd[f] = np.zeros(max_k, dtype=float)

    for res in results:
        for f in RES_FIELDS:
            res_line = np.array(res[f])
            rd[f] += res_line

    for f in RES_FIELDS:
        rd[f] /= len(SEEDS)
        rd[f] = rd[f].tolist()

    return rd


if __name__ == "__main__":
    config = cfg.Configuration(f"./config")

    # Initialize Evaluation Configuration
    for k, v in USE_CONFIG.items():
        globals()[k] = v

    # Configurations
    ds_config = config.ds_config()
    em_config = config.em_config(DATASET)
    ev_config = config.ev_config(DATASET)

    os.makedirs(f"./output/", exist_ok=True)

    # Runner
    runner = rs.RunStages(0, DATASET, None, None)
    graphs = runner.load_graphs(ds_config, NORMALIZATION, HIDE_PERC)

    for m in METHODS:
        print(f" <<< === {m} === >>>")

        method_results = []
        for s in SEEDS:
            # Learn embedding
            runner = rs.RunStages(s, DATASET, m, "link_prediction")
            runner.set_graphs(*graphs)
            runner.learn_embedding(em_config, DIMENSION)

            # Evaluate embedding
            result = runner.evaluate_embedding(ev_config)
            method_results.append(result)

        # Save Results
        results = process_results(method_results, ev_config["link_prediction"]["max_k"])
        with open(f"./output/{m}.txt", "w+") as fp:
            json.dump(results, fp)

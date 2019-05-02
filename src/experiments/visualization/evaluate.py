import sys
sys.path.insert(0, "../../")

import matplotlib.pyplot as plt
import run.configurations as cfg
import run.run_stages as rs
import os

# === DATASET CONFIGURATIONS ===
BUGZILLA_CONFIG = {
    "DATASET": "UW-bugzilla",
    "NORMALIZATION": None,
    "DIMENSIONS": 5,
    "SEED": 42
}

ZACHARY_KARATE_CONFIG = {
    "DATASET": "UW-zachary_karate",
    "NORMALIZATION": "log_global_norm",
    "DIMENSION": 2,
    "SEED": 42
}

# === SELECT WHICH DATASET AND METHODS ===
USE_CONFIG = ZACHARY_KARATE_CONFIG
METHODS = ["laplacian_eigenmaps", "deep_walk", "node2vec", "LINE", "SDNE"]

# === EVALUATION ===
if __name__ == "__main__":
    config = cfg.Configuration(f"./config")
    ds_config = config.ds_config()

    # Initialize Evaluation Configuration
    for k, v in USE_CONFIG.items():
        globals()[k] = v

    for m in METHODS:
        plt.figure(figsize=(6, 4.5), dpi=200)

        # Stage configurations
        em_config = config.em_config(DATASET)
        ev_config = config.ev_config(DATASET)

        # Learn embedding
        runner = rs.RunStages(SEED, DATASET, m, "visualization")
        runner.load_graphs(ds_config, NORMALIZATION)
        runner.learn_embedding(em_config, DIMENSION)

        # Evaluate embedding
        os.makedirs(f"./output/", exist_ok=True)
        runner.evaluate_embedding(ev_config, f"./output/{m}")

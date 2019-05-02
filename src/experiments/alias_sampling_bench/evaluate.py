import sys
sys.path.insert(0, "../../")

import embeddings.networks.alias_sampling as als
import numpy as np
import json
import time
import os

DRAWS = 10000
SEEDS = [42, 24, 31, 9, 8, 96, 21, 19, 6, 23, 2, 10, 22, 14, 5, 84, 34, 17, 51, 91]
CLASSES = [3, 5, 7, 10, 20, 50, 100, 200, 500, 1000]

if __name__ == "__main__":
    np_results = []
    als_results = []

    for d in CLASSES:
        np_class_results = []
        als_class_results = []
        preproc_times = []

        for s in SEEDS:
            # Generate probbilities
            np.random.seed(s)
            probs = np.random.random_sample(d)
            probs = probs / np.sum(probs)

            # Alias Sampling Preprocessing
            preproc_als = als.AliasSampling(probs, probs)

            # Numpy Random Choice
            list = []
            start = time.time()
            for i in range(DRAWS):
                sample = np.random.choice(probs, 1, p=probs)
                list.append(sample)
            np_class_results.append((time.time() - start))
            print(list)

            # Alias Sampling Choice
            list = []
            start = time.time()
            for i in range(DRAWS):
                sample = preproc_als.sample(1)
                list.append(sample)
            als_class_results.append((time.time() - start))
            print(list)

        np_results.append({"classes": d, "draw_time": np.mean(np.array(np_class_results))})
        als_results.append({"classes": d, "draw_time": np.mean(np.array(als_class_results))})

    # Save Results
    os.makedirs(f"./output/", exist_ok=True)
    with open(f"./output/np_random_choice.txt", "w+") as fp:
        json.dump(np_results, fp)

    with open(f"./output/alias_sampling.txt", "w+") as fp:
        json.dump(als_results, fp)

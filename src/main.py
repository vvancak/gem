import run.configurations as cfg
import run.run_stages as rs
import datetime as dt
import numpy as np
import argparse
import os

EMBED_DIR = "embeddings"
EVAL_DIR = "evaluations"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    # Directories
    parser.add_argument("--output", default="../output", type=str, help="Output directory")
    parser.add_argument("--config", default="../config", type=str, help="Configurations directory")

    # Main configurations
    parser.add_argument("--dataset", default="UW-zachary_karate", type=str, help="DataSet (from ds_config.json)")
    parser.add_argument("--embed_method", default="deep_walk", type=str,
                        help="Graph Embedding Method (from em_config.json)")
    parser.add_argument("--eval_method", default="visualization", type=str,
                        help="Evaluation Method (from ev_config.json)")

    # Specific configurations
    parser.add_argument("--embed_dim", default=2, type=int, help="Dimension of the created embedding")
    parser.add_argument("--force_learn", default=True, type=bool, help="Force the learning even if [embed_file] exists")
    parser.add_argument("--init_norm", default="log_global_norm", type=str,
                        help="Graph Normalization from init_norms.py (None if not required)")
    parser.add_argument("--hide_edges", default=None, type=int,
                        help="Hide x% of the graph's edges (None if not required)")
    parser.add_argument("--embed_file", default="out_embed", type=str,
                        help="Learned embedding filename (None if not required)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Run Managers
    runner = rs.RunStages(args.seed, args.dataset, args.embed_method, args.eval_method)
    config = cfg.Configuration(args.config)

    # Stage configurations
    ds_config = config.ds_config()
    em_config = config.em_config(args.dataset)
    ev_config = config.ev_config(args.dataset)

    # Graph
    print(f" <<< === GRAPHS === >>> ")
    runner.load_graphs(ds_config, args.init_norm, args.hide_edges)

    # Embedding
    print(f" <<< === EMBEDDING === >>> ")

    # Embedding Storing file not specified
    if not args.embed_file:
        runner.learn_embedding(em_config, args.embed_dim)

    else:
        embed_dir = f"{args.output}/{EMBED_DIR}"
        embed_file_path = f"{args.output}/{EMBED_DIR}/{args.embed_file}"
        if os.path.exists(f"{embed_file_path}.txt") and not args.force_learn:
            # File does exist and no force-learn
            runner.load_embedding(em_config, embed_file_path, args.embed_dim)
        else:
            # Learn & Store (default)
            runner.learn_embedding(em_config, args.embed_dim)

            os.makedirs(embed_dir, exist_ok=True)
            runner.store_embedding(embed_file_path)

    # Evaluation
    if args.eval_method:
        print(f" <<< === EVALUATION === >>>")
        timestamp = dt.datetime.now().strftime("%d-%H-%M-%S")
        eval_file = f"{args.embed_method}#{timestamp}"
        eval_dir = f"{args.output}/{EVAL_DIR}/{args.dataset}/{args.eval_method}"

        os.makedirs(eval_dir, exist_ok=True)
        runner.evaluate_embedding(ev_config, f"{eval_dir}/{eval_file}")

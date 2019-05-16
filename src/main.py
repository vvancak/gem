import run.configurations as cfg
import run.print_headers as ph
import run.run_stages as rs
import datetime as dt
import argparse
import os

TIMESTAMP = dt.datetime.now().strftime("%d-%H-%M")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    # 1. Directories
    parser.add_argument("--output", default="../output", type=str, help="Output directory")
    parser.add_argument("--config", default="../config", type=str, help="Configurations directory")

    # 2. Dataset
    parser.add_argument("--dataset", default="UW-zachary_karate", type=str, help="DataSet (from ds_config.json)")
    parser.add_argument("--init_norm", default=None, type=str, help="Graph Normalization from init_norms.py (None if not required)")
    parser.add_argument("--hide_edges", default=None, type=int, help="Hide x% of the graph's edges (None if not required)")

    # 3. Embeddings
    parser.add_argument("--embed_method", default="deep_walk", type=str, help="Graph Embedding Method (from em_config.json)")
    parser.add_argument("--embed_dim", default=2, type=int, help="Dimension of the created embedding")

    parser.add_argument("--embed_file", default=None, type=str, help="embedding file for loading embeddings")
    parser.add_argument("--embed_dir", default="embeddings", type=str, help="embedding directory within the output directory")

    # 4. Evaluations
    parser.add_argument("--eval_method", default=None, type=str, help="Evaluation Method (from ev_config.json)")
    parser.add_argument("--eval_dir", default="evaluations", type=str, help="evaluation directory within the output directory")

    # 5. Enlarge
    parser.add_argument("--add_edges", default=None, type=int, help="Add x% of the graph's edges (None if not required)")
    parser.add_argument("--enlarge_dir", default="enlarged", type=str, help="enlarged graph directory within the output directory")

    return parser.parse_args()


def graphs(runner: rs.RunStages, config: cfg.Configuration, args):
    print(f" <<< === GRAPHS === >>> ")
    ds_config = config.ds_config()
    runner.load_graphs(ds_config, args.init_norm, args.hide_edges)


def embeddings(runner: rs.RunStages, config: cfg.Configuration, args):
    print(f" <<< === EMBEDDING === >>> ")
    em_config = config.em_config(args.dataset)

    # Ensure we have directories
    embed_file_path = f"{args.output}/{args.embed_dir}/{args.dataset}"
    os.makedirs(embed_file_path, exist_ok=True)

    # Check if can load
    if args.embed_file:
        embed_file = f"{embed_file_path}/{args.embed_file}"
        if not os.path.exists(embed_file):
            print(f"{ph.ERROR} {args.embed_file} does not exist within {embed_file_path}")
            exit(1)
        runner.load_embedding(em_config, embed_file, args.embed_dim)

    else:
        # Or learn
        runner.learn_embedding(em_config, args.embed_dim)
        # And store
        embed_file = f"{args.embed_method}#{TIMESTAMP}.csv"
        runner.store_embedding(f"{embed_file_path}/{embed_file}")


def evaluations(runner: rs.RunStages, config: cfg.Configuration, args):
    print(f" <<< === EVALUATION === >>>")
    if args.eval_method:
        ev_config = config.ev_config(args.dataset)
        eval_file = f"{args.eval_method}#{args.embed_method}#{TIMESTAMP}"  # Not .txt, cause vis is .png
        eval_dir = f"{args.output}/{args.eval_dir}/{args.dataset}"

        os.makedirs(eval_dir, exist_ok=True)
        runner.evaluate_embedding(ev_config, f"{eval_dir}/{eval_file}")

    else:
        print(f"{ph.INFO} No method specified, skipping")


def enlarge(runner: rs.RunStages, config: cfg.Configuration, args):
    print(f" <<< === ENLARGE === >>> ")
    if args.add_edges:
        enlarge_file = f"{args.embed_method}#{TIMESTAMP}.csv"
        enlarge_dir = f"{args.output}/{args.enlarge_dir}/{args.dataset}"

        os.makedirs(enlarge_dir, exist_ok=True)
        runner.enlarge_graph(args.add_edges, f"{enlarge_dir}/{enlarge_file}")
    else:
        print(f"{ph.INFO} Not specified, wont add edges")


if __name__ == "__main__":
    args = parse_arguments()

    # Run Managers
    runner = rs.RunStages(args.seed, args.dataset, args.embed_method, args.eval_method)
    config = cfg.Configuration(args.config)

    # Run Stages
    graphs(runner, config, args)
    embeddings(runner, config, args)
    evaluations(runner, config, args)
    enlarge(runner, config, args)

    print(f"{ph.OK} Done.")

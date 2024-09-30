import argparse
import importlib
import json
import os

import orbax
from ml_collections import ConfigDict

from utilities.utils import dot_key_dict_to_nested_dicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    parser.add_argument("-g", type=int, default=0)
    parser.add_argument("--evaluator_class", type=str, default="OnlineEvaluator")
    parser.add_argument("--num_eval_envs", type=int, default=10)
    parser.add_argument("--eval_n_trajs", type=int, default=10)
    parser.add_argument("--eval_env_seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.g < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.g)

    with open(os.path.join(args.log_dir, "variant.json"), "r") as f:
        variant = json.load(f)

    config = dot_key_dict_to_nested_dicts(variant)
    config = ConfigDict(config)

    # rewrite configs
    config.evaluator_class = args.evaluator_class
    config.num_eval_envs = args.num_eval_envs
    config.eval_n_trajs = args.eval_n_trajs
    config.eval_env_seed = args.eval_env_seed
    config.eval_batch_size = args.eval_batch_size

    trainer = getattr(importlib.import_module("diffuser.trainer"), config.trainer)(
        config, use_absl=False
    )
    trainer._setup()

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    target = {"agent_states": trainer._agent.train_states}
    for epoch in args.epochs:
        ckpt_path = os.path.join(args.log_dir, f"checkpoints/model_{epoch}")
        restored = orbax_checkpointer.restore(ckpt_path, item=target)
        eval_params = {
            key: restored["agent_states"][key].params_ema or restored["agent_states"][key].params
            for key in trainer._agent.model_keys
        }
        trainer._evaluator.update_params(eval_params)
        metrics = trainer._evaluator.evaluate(epoch)
        print(f"\033[92m Epoch {epoch}: {metrics} \033[00m\n")


if __name__ == "__main__":
    main()

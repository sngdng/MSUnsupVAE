import argparse
import logging
import os
from collections import Counter
from uuid import uuid4
from pathlib import Path
from typing import Union, Set, Tuple, Dict

import wandb
import torch
from omegaconf import OmegaConf
from d2t.utils import (
            WarningsFilter,
            seed_everything,
            ModelSummary,
            Mode,
            VAEModel,
        )
from d2t.eval.vae_evaluator import VAEvaluator
from d2t.data.datasets import get_dataset_class
from d2t.model import DT7ImDisStyleVAE
from d2t.data.tokenization_t5 import VAET5Tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model in d2t & t2d')
    parser.add_argument('-c', '--config',
                        help='Path to the config')
    parser.add_argument('-w', '--weight',
                        help="Path to model's weight")
    parser.add_argument('-sx', '--style_x', type=int, default=10,
                        help="Number of s_x to be sampled from posterior")

    args = parser.parse_args()

    return args


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    project_dir = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(project_dir / args.config)
    vae_model = VAEModel(conf.vae.model)

    import wandb
    import yaml
    # Set WandB API key & config dir
    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ['WANDB_API_KEY'] = '1ae76601c5643dd4b94d5764764a67c107cbb14a' ## PUT YOUR WANDB_API_KEY ##
    os.environ["WANDB_CONFIG_DIR"] = "."
    os.environ["WANDB_CACHE_DIR"] = "."
    os.environ["WANDB_DIR"] = "."
    wandb.init(project='ms_unsup_vae_t5',
               entity="sduong", ## ENTITY ##
               config=yaml.safe_load(open(project_dir / args.config)),
               dir=".",
               settings=wandb.Settings(start_method="thread"))

    data_dir = project_dir / "data"
    datasets = {
        split: get_dataset_class(conf.dataset_name)(
            data_dir=data_dir, split=split
        )
        for split in get_dataset_class(conf.dataset_name).splits if split.startswith('test')
    }
    vae_model = VAEModel(conf.vae.model)
    model = DT7ImDisStyleVAE.from_pretrained(
            conf.model,
            nb_formats=len(get_dataset_class(conf.dataset_name).id2format),
            s_x_dim=conf.vae.s_x_dim,
            specify_target_with_prefix=conf.specify_target_with_prefix,
            generate_text_token_id=None,
            generate_data_token_id=None,
            reg_loss=conf.vae.reg,
            use_style_token=conf.vae.use_style_token,
        )
    # load tokenizer
    tokenizer = VAET5Tokenizer.from_pretrained(
        conf.model,
        use_style_token=conf.vae.use_style_token
    )
    # extend embedding matrices to include new tokens
    model.resize_token_embeddings(len(tokenizer))
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
    model.to(device)
    model.eval()
    summary = ModelSummary(model, mode="top")
    logging.info(f"\n{summary}")
    
    evaluator = VAEvaluator(
        mode=Mode(conf.mode),
        datasets=datasets,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=conf.batch_size_val,
        num_beams_t2d=conf.num_beams_t2d,
        num_beams_d2t=conf.num_beams_d2t,
        log_path=project_dir,
        num_sx=args.style_x,
        dim_sx=conf.vae.s_x_dim
    )
    evaluator.evaluate_and_log()

    wandb.finish()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

import argparse
import datetime
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Union, Set, Tuple, Dict

import wandb
import torch
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from datasets import load_metric, Metric
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    default_data_collator,
    PreTrainedTokenizer,
    PreTrainedModel,
    T5ForConditionalGeneration,
    AutoTokenizer
)

from d2t.model import DT7ImDisStyleVAE, DT7NonVAE
from d2t.data.datasets import Seq2seqDataset, get_dataset_class
from d2t.data.tokenization_t5 import VAET5Tokenizer
from d2t.data.formatting import Example, DataFormat, STYLE_TOKEN, construct_all_prefixes
from d2t.eval.evaluator import Evaluator
from d2t.eval.utils import get_precision_recall_f1, compute_meteor_score
from d2t.eval.metrics.sembleu_score import NgramInstance, corpus_bleu
from d2t.utils import MyLogger, Mode, ModelSummary, VAEModel


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model in d2t & t2d')
    parser.add_argument('-c', '--config',
                        help='Path to the config')
    parser.add_argument('-w', '--weight', required=True,
                        help="Path to model's weight")

    args = parser.parse_args()

    return args


def main(args):
    project_dir = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(project_dir / args.config)
    # multi-GPU handler
    vae_model = VAEModel(conf.vae.model)
    accelerator_kwargs = []
    if vae_model == VAEModel.im_dis_style_vae:
        # for the StyleVAE, we don't use all parameters before each backward call (since we compute
        # either vae_s_x or vae_s_y), so this is necessary (https://pytorch.org/docs/stable/notes/ddp.html)
        accelerator_kwargs = [
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    accelerator = Accelerator(kwargs_handlers=accelerator_kwargs)
    timestamp = datetime.datetime.today().strftime("%m%d%H%M%S")
    run_name = f"{timestamp}-{conf.mode}-{conf.model}-{conf.vae.model}"
    if conf.mode == "both_unsup":
        run_name += f"-{conf.generate_method}"
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
    # prepare model (todo: put parameters in model config and load from_config?)
    if vae_model == VAEModel.non_vae:
        model = DT7NonVAE.from_pretrained(
            conf.model,
            specify_target_with_prefix=conf.specify_target_with_prefix,
            generate_text_token_id=generate_text_tok_id,
            generate_data_token_id=generate_data_tok_id,
        )
    elif vae_model == VAEModel.im_dis_style_vae:
        model = DT7ImDisStyleVAE.from_pretrained(
            conf.model,
            nb_formats=len(train_dataset.id2format),
            s_x_dim=conf.vae.s_x_dim,
            specify_target_with_prefix=conf.specify_target_with_prefix,
            generate_text_token_id=None,
            generate_data_token_id=None,
            reg_loss=conf.vae.reg,
            use_style_token=conf.vae.use_style_token,
        )
    else:
        raise ValueError
        
    # load tokenizer
    tokenizer = VAET5Tokenizer.from_pretrained(
        conf.model,
        use_style_token=conf.vae.use_style_token
    )
    # extend embedding matrices to include new tokens
    model.resize_token_embeddings(len(tokenizer))
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
    ddp_model = accelerator.prepare(model)
    summary = ModelSummary(model, mode="top")
    logging.info(f"\n{summary}")
    
    evaluator = Evaluator(
        run_name=run_name,
        mode=Mode(conf.mode),
        datasets=datasets,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ddp_model=ddp_model,
        batch_size=conf.batch_size_val,
        num_beams_t2d=conf.num_beams_t2d,
        num_beams_d2t=conf.num_beams_d2t,
        log_path=project_dir / f"models/{run_name}",
        checkpoints=conf.checkpoints,
        limit_samples=100 if conf.fast_dev_run else False,
        is_vae=(vae_model != VAEModel.non_vae),
        do_validation=False
    )
    for split in datasets.keys():
        evaluator.evaluate_and_log(-1, split=split)

    wandb.finish()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

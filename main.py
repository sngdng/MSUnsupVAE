import argparse
import datetime
import logging
import sys
import os
from pathlib import Path

import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from d2t.data.datasets import get_dataset_class
from d2t.data.formatting import (
    DataFormat,
    GENERATE_TEXT_TOKEN,
    STYLE_TOKEN,
)
from d2t.data.tokenization_t5 import VAET5Tokenizer
from d2t.eval.evaluator import Evaluator
from d2t.model import DT7NonVAE, DT7ImDisStyleVAE
from d2t.trainer import Seq2seqTrainer
from d2t.utils import (
    WarningsFilter,
    seed_everything,
    ModelSummary,
    Mode,
    VAEModel,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-c', '--config',
                        help='Path to the config')

    args = parser.parse_args()

    return args


def main(timestamp: str, config_path: str):
    # Load config
    project_dir = Path(__file__).resolve().parents[0]

    conf = OmegaConf.load(project_dir / config_path)

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

    if accelerator.is_local_main_process:
        import yaml
        # Set WandB API key & config dir
        os.environ["WANDB_START_METHOD"] = "thread"
        os.environ['WANDB_API_KEY'] = "1ae76601c5643dd4b94d5764764a67c107cbb14a" ## PUT YOUR WANDB_API_KEY ##
        os.environ["WANDB_CONFIG_DIR"] = "."
        os.environ["WANDB_CACHE_DIR"] = "."
        os.environ["WANDB_DIR"] = "."
        wandb.init(project='ms_unsup_vae_t5',
                   entity="sduong", ## ENTITY ##
                   config=yaml.safe_load(open(project_dir / config_path)),
                   dir=".",
                   settings=wandb.Settings(start_method="thread"))
        # Log code directory as artifact
        code_artifact = wandb.Artifact('d2t', 'code')
        code_artifact.add_dir(str(project_dir / "d2t"))
        wandb.log_artifact(code_artifact)

    # format logging
    logging.basicConfig(
        format="%(process)d %(asctime)s %(message)s",
        datefmt="[%H:%M:%S]",
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
    )

    # complete and print conf, with a specific run name
    use_loggers = accelerator.is_local_main_process and not conf.fast_dev_run
    conf.use_fp16 = accelerator.use_fp16
    conf.num_processes = accelerator.num_processes
    logging.info(OmegaConf.to_yaml(conf))
    run_name = f"{timestamp}-{conf.mode}-{conf.model}-{conf.vae.model}"
    if conf.mode == "both_unsup":
        run_name += f"-{conf.generate_method}"

    # seed everything
    seed_everything(conf.seed)

    logging.info(f"run_name: {run_name}\n")
    if use_loggers:
        tb_writer = SummaryWriter(log_dir=str(project_dir / f"models/{run_name}"))
    else:
        tb_writer = None

    # load tokenizer
    tokenizer = VAET5Tokenizer.from_pretrained(
        conf.model,
        use_style_token=conf.vae.use_style_token
    )

    # load data
    data_dir = project_dir / "data"
    datasets = {
        split: get_dataset_class(conf.dataset_name)(
            data_dir=data_dir, split=split, tokenizer=tokenizer, accelerator=accelerator
        )
        for split in get_dataset_class(conf.dataset_name).splits
    }
    train_dataset = datasets["train"]

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

    # extend embedding matrices to include new tokens
    model.resize_token_embeddings(len(tokenizer))
    summary = ModelSummary(model, mode="top")
    logging.info(f"\n{summary}")

    trainer = Seq2seqTrainer(
        model=model,
        mode=Mode(conf.mode),
        vae_model=vae_model,
        vae_beta=conf.vae.beta,
        beta_n_cycle=conf.vae.beta_n_cycle,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        accelerator=accelerator,
        learning_rate=conf.lr * conf.num_processes,
        lr_scheduler=conf.lr_scheduler,
        batch_size=conf.batch_size_train,
        noise_fn=conf.sample_noise_fun,
        generate_method=conf.generate_method,
        num_epochs=conf.epochs,
        tensorboard_writer=tb_writer,
        log_path=project_dir / f"models/{run_name}",
        log_every_n_steps=conf.log_every_n_steps,
        max_grad_norm=conf.max_grad_norm,
        max_training_steps=3 if conf.fast_dev_run else -1,
    )

    if accelerator.is_local_main_process:
        evaluator = Evaluator(
            run_name=run_name,
            mode=Mode(conf.mode),
            datasets=datasets,
            tokenizer=tokenizer,
            accelerator=accelerator,
            ddp_model=trainer.ddp_model,
            batch_size=conf.batch_size_val,
            num_beams_t2d=conf.num_beams_t2d,
            num_beams_d2t=conf.num_beams_d2t,
            log_path=project_dir / f"models/{run_name}",
            checkpoints=conf.checkpoints,
            tensorboard_writer=tb_writer,
            limit_samples=100 if conf.fast_dev_run else False,
            is_vae=(vae_model != VAEModel.non_vae),
            do_validation=conf.do_validation
        )
        trainer.set_evaluator(evaluator)
    trainer.train()
    
    if accelerator.is_local_main_process:
        wandb.finish()


if __name__ == "__main__":
    # filter out some hdfs warnings (from 3rd party python libraries)
    sys.stdout = WarningsFilter(sys.stdout)
    sys.stderr = WarningsFilter(sys.stderr)
    args = parse_arguments()
    main(timestamp=datetime.datetime.today().strftime("%m%d%H%M%S"), config_path=args.config)

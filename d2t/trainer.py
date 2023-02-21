import logging
import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    get_scheduler,
    default_data_collator,
    PreTrainedTokenizer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from accelerate import Accelerator

from d2t.data.datasets import Seq2seqDataset
from d2t.data.formatting import add_target_prefix, add_style_prefix, construct_all_prefixes
from d2t.data.noise import existing_noise_functions
from d2t.eval.evaluator import Evaluator
from d2t.model import (
    DT7ModelOutput,
    VariationalT5EncoderOutput,
)
from d2t.utils import MyLogger, Mode, frange_cycle_zero_linear, VAEModel


class Seq2seqTrainer:
    # to be set after trainer init (we need to create Trainer with accelerator first)
    evaluator: Evaluator

    def __init__(
        self,
        model,
        mode: Mode,
        vae_model: VAEModel,
        vae_beta: float,
        beta_n_cycle: int,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Seq2seqDataset,
        accelerator: Accelerator,
        learning_rate: float,
        lr_scheduler: str,
        batch_size: int,
        num_epochs: int,
        noise_fn: List[str],
        generate_method: str,
        tensorboard_writer: SummaryWriter,
        log_path: Path,
        log_every_n_steps: int,
        max_grad_norm: float,
        max_training_steps: int = -1,
    ):
        self.mode = mode
        self.tokenizer = tokenizer

        # training
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )
        # prepare model and data for multi gpu training (if necessary)
        self.accelerator = accelerator
        self.ddp_model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.id2format = train_dataloader.dataset.id2format
        if getattr(model.encoder, 's_x_dim', None):
            self.sx_dim = model.encoder.s_x_dim

        # training parameters
        self.batch_size = batch_size
        # max training steps per epoch
        if max_training_steps > 0:
            # stop early for testing purposes
            self.max_training_steps = max_training_steps
        else:
            self.max_training_steps = len(self.train_dataloader)
        self.num_training_steps = num_epochs * self.max_training_steps
        self.num_epochs = num_epochs
        if lr_scheduler == "cosine_with_restarts":
            self.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.num_training_steps,
                num_cycles=4,
            )
        else:
            self.lr_scheduler = get_scheduler(
                name=lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.num_training_steps,
            )
        self.max_seq_length = train_dataset.max_seq_length
        self.max_grad_norm = max_grad_norm
        self.noise_functions = noise_fn
        self.generate_method = generate_method
        # VAE specific code
        self.vae_model = vae_model
        self.use_vae = (vae_model != VAEModel.non_vae)
        if self.use_vae:
            self.use_cyclical_beta_schedule = beta_n_cycle > -1
            if self.use_cyclical_beta_schedule:
                self.betas = frange_cycle_zero_linear(self.num_training_steps, beta_n_cycle)
            else:
                # constant beta coefficient -> specify once to the model
                model.beta = vae_beta

        # logging
        self.log_path = log_path
        self.log_every_n_steps = log_every_n_steps
        self.logger = MyLogger(
            tensorboard_writer=tensorboard_writer,
            log_every_n_steps=log_every_n_steps,
            accelerator=accelerator,
            use_loggers=True,
        )

        # compute prefixes once and for all
        self.prefixes = construct_all_prefixes(tokenizer)

    def set_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def predict(self, input_ids: torch.Tensor, target: str, **kwargs):
        model = self.accelerator.unwrap_model(self.ddp_model)
        prediction_ids = model.generate_with_target(
            input_ids=input_ids,
            target=target,
            prefixes=self.prefixes,
            max_seq_length=self.max_seq_length,
            method=self.generate_method,
            **kwargs
        )
        # multi-GPU: no need to gather predictions across processes yet, since the
        # predictions are to be used in training (gathering is down after the loss is computed)
        return prediction_ids


    def teach_model_one_step(
        self,
        input_ids: Optional[torch.Tensor],
        label_ids: Optional[torch.Tensor],
        target: Optional[str] = None,
        vae_latent_s: Optional[torch.Tensor] = None,
        retain_graph: bool = False,
        source: Optional[str] = None,
        format_data: Optional[torch.Tensor] = None
    ) -> DT7ModelOutput:
        """
        Run a forward pass in the model using input_ids, and backward the loss wrt label_ids.
        If necessary, append prefix to input_ids (to specify text/data target).

        Args:
            input_ids: input sequence (text/data tokenized batch, already on device).
                Should be None if encoder_outputs is given.
            label_ids: label (ground truth data/text as a tokenized sequence)
            target: 'text' or 'data', depending on the format of label sequences. Will
                determine the prefix to add to input_ids, or the token to specify to the decoder
            retain_graph: passed to the backward (default: False)
            source: None by default, otherwise 'text' or 'data' and passed to model forward
            vae_latent_s : reuse previously encoded latent code

        Returns:
            model_outputs

        """

        model = self.accelerator.unwrap_model(self.ddp_model)
        kwargs = {}  # model forward kwargs

        if input_ids is not None:
            if model.specify_target_with_prefix and target is not None:
                if format_data is not None:
                    assert target == 'data'
                    format_target = [self.id2format[f.item()] for f in format_data]
                    input_ids = add_target_prefix(
                        input_ids=input_ids,
                        target=format_target,
                        prefixes=self.prefixes
                    )
                else:
                    # add the prefix "generate data/text" to the input
                    # (if model.specify_target_with_prefix=False, the model handles this itself
                    # and uses the right decoder_start_token_id in the forward)
                    input_ids = add_target_prefix(
                        input_ids=input_ids,
                        target=target,
                        prefixes=self.prefixes
                    )

            if source is not None:
                kwargs["source"] = source
            if getattr(model, "use_style_token", None):
                input_ids = add_style_prefix(
                    input_ids=input_ids, tokenizer=self.tokenizer
                )

            kwargs["input_ids"] = input_ids
            kwargs["attention_mask"] = self.get_att_mask(input_ids)
        
        if label_ids is not None:
            kwargs["target"] = target
            kwargs["labels"] = label_ids

        if vae_latent_s is not None:
            kwargs["encoder_outputs"] = VariationalT5EncoderOutput(
                vae_latent_s=vae_latent_s
            )

        self.ddp_model.train()
        outputs = self.ddp_model(**kwargs)
        if label_ids is None:
            loss = model.beta * outputs.reg_loss / 2.
        else:
            loss = outputs.loss

        # we call loss.backward() here to free GPU memory for the next steps
        # -> computed gradients are kept in the leaf variables (the parameters)
        # -> but the computational data is removed elsewhere
        # -> equivalent to calling backward on the sum of the losses, since gradients
        #   are added until we call .zero_grad()
        self.accelerator.backward(loss, retain_graph=retain_graph)
        return outputs

    def compute_loss_unsup_dis_style_vae(
        self,
        text_ids: torch.Tensor,
        data_ids: torch.Tensor,
        format_data: torch.Tensor,
    ):
        if len(self.id2format) == 1:
            syn_data_s = torch.ones((format_data.size(0), 1)).to(data_ids.device)
        else:
            syn_data_s = torch.nn.functional.one_hot(format_data, num_classes=len(self.id2format)).to(data_ids.device)
        # -- AUTO DENOISING LOSS --
        noisy_text_ids = self.get_noisy_inputs(text_ids, target_format_data=format_data)
        noisy_data_ids = self.get_noisy_inputs(data_ids, format_data=format_data)
        # log p(x|c,s_x) - KL(q(s_x|x_noisy) || p(s_x))
        text_style_outputs = self.teach_model_one_step(
            input_ids=text_ids,
            label_ids=None,
            source="text",
            retain_graph=True
        )
        text_outputs = self.teach_model_one_step(
            noisy_text_ids, text_ids,
            target="text",
            vae_latent_s=text_style_outputs.vae_latent_s,
        )
        data_outputs = self.teach_model_one_step(
            noisy_data_ids, data_ids,
            target="data",
            vae_latent_s=syn_data_s,
        )

        # -- CYCLE LOSS --
        # sample from normal gaussian
        #text_prior = torch.distributions.Normal(
        #    loc=torch.zeros((data_ids.size(0), self.sx_dim)).to(data_ids.device),
        #    scale=torch.ones((data_ids.size(0), self.sx_dim)).to(data_ids.device),
        #)
        #syn_text_s = text_prior.sample()
        syn_text_s = torch.zeros((data_ids.size(0), self.sx_dim)).to(data_ids.device)
        syn_data_ids = self.predict(input_ids=text_ids, target="data", vae_latent_s=syn_data_s)
        syn_text_ids = self.predict(input_ids=data_ids, target="text", vae_latent_s=syn_text_s)

        # Data2Text (y_hat -> x)
        # we need retain_graph=True since in the next step we'll backward through vae_c again
        d2t_style_outputs = self.teach_model_one_step(
            input_ids=text_ids,
            label_ids=None,
            source="text",
            retain_graph=True
        )  # - KL(q(s_x|x) || p(s_x)), with s_x~q(s_x|c,x)
        d2t_outputs = self.teach_model_one_step(
            input_ids=syn_data_ids,
            label_ids=text_ids,
            target="text",
            vae_latent_s=d2t_style_outputs.vae_latent_s,
        )  # log p(x|c,s_x)

        # Text2Data (x_hat -> y)
        # we need retain_graph=True since in the next step we'll backward through vae_c again
        t2d_outputs = self.teach_model_one_step(
            input_ids=syn_text_ids,
            label_ids=data_ids,
            target="data",
            vae_latent_s=syn_data_s,
        )  # log p(y|c,s_y)

        return {
            "text": text_outputs,
            "data": data_outputs,
            "t2d": t2d_outputs,
            "d2t_style": d2t_style_outputs,
            "d2t": d2t_outputs
        }
    
    def compute_loss_unsup_non_vae(
        self, text_ids: torch.Tensor, data_ids: torch.Tensor, format_data: torch.Tensor
    ):
        # -- auto loss (denoising auto-encoding)
        noisy_text_ids = self.get_noisy_inputs(text_ids, target_format_data=format_data)
        noisy_data_ids = self.get_noisy_inputs(data_ids, format_data=format_data)
        text_outputs = self.teach_model_one_step(
            noisy_text_ids, text_ids, target="text"
        )
        data_outputs = self.teach_model_one_step(
            noisy_data_ids, data_ids, target="data"
        )

        # -- cycle loss
        syn_data_ids = self.predict(input_ids=text_ids, target="data")
        syn_text_ids = self.predict(input_ids=data_ids, target="text")
        d2t_outputs = self.teach_model_one_step(syn_data_ids, text_ids, target="text")
        t2d_outputs = self.teach_model_one_step(syn_text_ids, data_ids, target="data")

        for out in [text_outputs, data_outputs, d2t_outputs, t2d_outputs]:
            # since we have a non VAE model, reg_loss is None and loss=recon_loss
            assert out.reg_loss is None
        return {
            "text": text_outputs,
            "data": data_outputs,
            "t2d": t2d_outputs,
            "d2t": d2t_outputs,
        }
    
    def compute_loss_unsup_mf_non_vae(
        self, text_ids: torch.Tensor, data_ids: torch.Tensor, format_data: torch.Tensor
    ):
        # -- auto loss (denoising auto-encoding)
        noisy_text_ids = self.get_noisy_inputs(text_ids, target_format_data=format_data)
        noisy_data_ids = self.get_noisy_inputs(data_ids, format_data=format_data)
        text_outputs = self.teach_model_one_step(
            noisy_text_ids, text_ids, target="text"
        )
        data_outputs = self.teach_model_one_step(
            noisy_data_ids, data_ids, target="data",
            format_data=format_data
        )

        # -- cycle loss
        format_target = [self.id2format[f.item()] for f in format_data]
        syn_data_ids = self.predict(input_ids=text_ids, target="data", format_target=format_target)
        syn_text_ids = self.predict(input_ids=data_ids, target="text")
        d2t_outputs = self.teach_model_one_step(syn_data_ids, text_ids, target="text")
        t2d_outputs = self.teach_model_one_step(syn_text_ids, data_ids, target="data", format_data=format_data)

        for out in [text_outputs, data_outputs, d2t_outputs, t2d_outputs]:
            # since we have a non VAE model, reg_loss is None and loss=recon_loss
            assert out.reg_loss is None
        return {
            "text": text_outputs,
            "data": data_outputs,
            "t2d": t2d_outputs,
            "d2t": d2t_outputs,
        }

    def compute_loss_unsup_vae_single(
        self, text_ids: torch.Tensor, data_ids: torch.Tensor
    ):
        # -- auto loss (regular VAE)
        text_outputs = self.teach_model_one_step(text_ids, text_ids, target="text")
        data_outputs = self.teach_model_one_step(data_ids, data_ids, target="data")

        # -- cycle loss (single)
        syn_data_ids = self.predict(input_ids=text_ids, target="data")
        syn_text_ids = self.predict(input_ids=data_ids, target="text")
        d2t_outputs = self.teach_model_one_step(syn_data_ids, text_ids, target="text")
        t2d_outputs = self.teach_model_one_step(syn_text_ids, data_ids, target="data")

        # total loss
        # loss = (
        #     text_outputs.loss + data_outputs.loss + d2t_outputs.loss + t2d_outputs.loss
        # )
        return {
            "text": text_outputs,
            "data": data_outputs,
            "t2d": t2d_outputs,
            "d2t": d2t_outputs,
        }

    def train(self):
        self.global_step = 0
        logging.info("Training...")
        logging.info(f"     num_epochs: {self.num_epochs}")

        for epoch in range(self.num_epochs):
            for batch in tqdm(
                self.train_dataloader,
                desc=f"[ep{epoch}]",
                disable=not self.accelerator.is_local_main_process,
            ):
                # stop training if a max number of steps was specified
                if self.global_step >= self.max_training_steps * (epoch + 1):
                    break

                # get batch data
                text_ids = batch["text_ids"]
                data_ids = batch["data_ids"]
                att_mask_text = batch["att_mask_text"]
                att_mask_data = batch["att_mask_data"]
                format_data = batch["format_data"]
                # simply make sure our get_att_mask method works
                assert (att_mask_data == self.get_att_mask(data_ids)).all()
                assert (att_mask_text == self.get_att_mask(text_ids)).all()

                # training step
                outputs = {}
                syn_text_ids = None  # synthetic text and datas
                syn_data_ids = None
                noisy_text_ids = None
                noisy_data_ids = None

                # select beta
                if self.use_vae and self.use_cyclical_beta_schedule:
                    model = self.accelerator.unwrap_model(self.ddp_model)
                    model.beta = self.betas[self.global_step]

                if self.mode == Mode.t2d:
                    outputs["t2d"] = self.teach_model_one_step(
                        input_ids=text_ids,
                        label_ids=data_ids,
                        target="data",
                    )
                elif self.mode == Mode.t2mfd:
                    outputs["t2d"] = self.teach_model_one_step(
                        input_ids=text_ids,
                        label_ids=data_ids,
                        target="data",
                        format_data=format_data
                    )
                elif self.mode == Mode.d2t:
                    outputs["d2t"] = self.teach_model_one_step(
                        input_ids=data_ids,
                        label_ids=text_ids,
                        target="text",
                    )
                elif self.mode == Mode.both_sup:
                    outputs["d2t"] = self.teach_model_one_step(
                        input_ids=data_ids,
                        label_ids=text_ids,
                        target="text",
                    )
                    outputs["t2d"] = self.teach_model_one_step(
                        input_ids=text_ids,
                        label_ids=data_ids,
                        target="data",
                    )
                elif self.mode == Mode.both_unsup:
                    # todo: make sure the predictions are correctly formatted, especially the attention mask
                    #   -> does it start with an unnecessary padding token?
                    if self.vae_model == VAEModel.non_vae:
                        outputs = self.compute_loss_unsup_non_vae(
                            text_ids=text_ids,
                            data_ids=data_ids,
                            format_data=format_data
                        )
                    elif self.vae_model == VAEModel.im_dis_style_vae:
                        outputs = self.compute_loss_unsup_dis_style_vae(
                            text_ids=text_ids,
                            data_ids=data_ids,
                            format_data=format_data,
                        )
                elif self.mode == Mode.both_unsup_mf:
                    outputs = self.compute_loss_unsup_mf_non_vae(
                        text_ids=text_ids,
                        data_ids=data_ids,
                        format_data=format_data,
                    )
                else:
                    raise ValueError

                # loss.backward has already been called (in teach_model_one_step)
                self.accelerator.clip_grad_norm_(
                    self.ddp_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.global_step += 1

                # log training info (metrics and text sequences)
                self.log_metrics(epoch, model_outputs=outputs)
                self.log_training_samples(
                    epoch,
                    text_ids=text_ids,
                    data_ids=data_ids,
                    syn_text_ids=syn_text_ids,
                    syn_data_ids=syn_data_ids,
                    noisy_text_ids=noisy_text_ids,
                    noisy_data_ids=noisy_data_ids,
                    t2d_logits=outputs["t2d"].logits if "t2d" in outputs else None,
                    d2t_logits=outputs["d2t"].logits if "d2t" in outputs else None,
                )

            # free GPU memory before eval
            outputs = {}
            # evaluate after each epoch (and save model checkpoint if necessary)
            self.evaluator.on_epoch_end(epoch)
            self.logger.send_current_logs()

        # evaluate on test set
        #   todo: remove when tuning hyperparameters, to make sure we don't overfit on test set
        self.evaluator.on_training_end()

    @staticmethod
    def get_att_mask(input_ids: torch.Tensor):
        # attention mask: 0 if it's a padding token, 1 otherwise
        # also type as input ids (tensor of integers)
        att_mask = (input_ids != 0).type_as(input_ids)
        return att_mask

    def get_noisy_inputs(
        self,
        input_ids: torch.Tensor,
        format_data: torch.Tensor = None,
        target_format_data: torch.Tensor = None
    ):
        # decode input ids
        seqs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # add noise to the texts/datas
        noisy_inputs = []
        for i, seq in enumerate(seqs):
            noise_fun_name = random.choice(self.noise_functions)
            noise_fun = existing_noise_functions[noise_fun_name]
            source_format = self.id2format[format_data[i].item()] if format_data is not None else "text"
            target_format = self.id2format[target_format_data[i].item()] if source_format == "text" else None
            noisy_seq, _ = noise_fun(seq, source_format=source_format, target_format=target_format)
            noisy_inputs.append(noisy_seq)

        # tokenize back
        batch_encoding = self.tokenizer(
            noisy_inputs,
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        batch_encoding = batch_encoding.to(input_ids.device)
        noisy_ids = batch_encoding.input_ids
        return noisy_ids

    def log_metrics(self, epoch: int, model_outputs: dict):
        if (
            self.accelerator.is_local_main_process
        ):
            metrics = {
                "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                "train/epoch": epoch,
            }
            for mode, outputs in model_outputs.items():
                # for each mode (t2d, d2t, text, ...), log our regular and vae metrics
                outputs = {k: v for k, v in outputs.__dict__.items() if v is not None}
                if "loss" in outputs:
                    metrics[f"train/loss_{mode}"] = outputs['loss'].item()
                if "recon_loss" in outputs:
                    metrics[f"train/recon_loss_{mode}"] = outputs['recon_loss'].item()
                if "reg_loss" in outputs and isinstance(outputs['reg_loss'], torch.Tensor):
                    metrics[f"train/reg_loss_{mode}"] = outputs['reg_loss'].item()
                if "reg_content_loss" in outputs and isinstance(outputs['reg_content_loss'], torch.Tensor):
                    metrics[f"train/reg_content_loss_{mode}"] = outputs['reg_content_loss'].item()

            # log vae_beta coeff if we have a VAE model
            model = self.accelerator.unwrap_model(self.ddp_model)
            if hasattr(model, "beta"):
                metrics["train/beta_t"] = model.beta

            self.logger.log_metrics(metrics)

    def log_training_samples(self, epoch: int, **kwargs):
        if (
            self.global_step % self.log_every_n_steps == 0 and self.accelerator.is_local_main_process
        ):
            # make missing predictions (e.g. in supervised mode, where we don't
            # need to generate fake samples)
            if kwargs["syn_data_ids"] is None:
                kwargs["syn_data_ids"] = self.predict(
                    input_ids=kwargs["text_ids"], target="data"
                )
            if kwargs["syn_text_ids"] is None:
                kwargs["syn_text_ids"] = self.predict(
                    input_ids=kwargs["data_ids"], target="text"
                )
            # convert logits (obtained from the forward call with teacher forcing)
            # into token id sequences
            if kwargs["t2d_logits"] is not None:
                t2d_logits = kwargs.pop("t2d_logits")
                kwargs["tf_data_ids"] = t2d_logits.argmax(dim=-1)
            if kwargs["d2t_logits"] is not None:
                d2t_logits = kwargs.pop("d2t_logits")
                kwargs["tf_text_ids"] = d2t_logits.argmax(dim=-1)

            # in the end we want:
            # training_samples = {"noisy_text": ["sentence", "sentence2", "sentence3"]}
            training_samples = {}
            for name, token_ids in kwargs.items():
                if token_ids is None:
                    continue
                # decode the tensor of token ids into a list of strings
                # (one per example of the batch)
                sentences = self.tokenizer.batch_decode(
                    token_ids, skip_special_tokens=True
                )
                if name.endswith("_ids"):
                    name = name[:-4]
                training_samples[name] = sentences

            # format log text
            log = f"[{self.global_step}] {', '.join(training_samples.keys())}\n"
            batch_size = len(training_samples["text"])
            batch_logs = ["" for _ in range(batch_size)]
            for sentences in training_samples.values():
                for i, s in enumerate(sentences):
                    # concatenate all sentences for the same batch example together
                    batch_logs[i] += f"{s}\n"
            log += "-\n".join(batch_logs)

            # save logs to disk
            logs_path = self.log_path / f"training/{epoch}.txt"
            self.logger.log_text(
                text=log,
                file_path=logs_path,
                folder_name="training",
                one_time_log=False,  # don't log to mlflow until the end of the epoch
            )
import copy
import warnings
import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import torch
from dataclasses import dataclass
from torch import nn
from torch.distributions import Normal, kl_divergence, Categorical
from torch.nn import CrossEntropyLoss
from transformers import T5PreTrainedModel, T5Config
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.t5.modeling_t5 import T5Stack, T5Block
from transformers.utils import logging

from d2t.data.formatting import add_target_prefix

logger = logging.get_logger(__name__)
# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
_HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@dataclass
class VariationalT5EncoderOutput(ModelOutput):
    """
    Same attributes as the BaseModelOutputWithPastAndCrossAttentions class (output of the regular
    encoder, T5Stack)

    And some attributes from the VAE encoder:
        q_phi
        vae_latent
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # variational posterior of style
    q_phi_s: Normal = None
    # the latent c sampled from q_phi_c after encoding
    vae_latent_s: torch.Tensor = None


@dataclass
class DT7ModelOutput(ModelOutput):
    """
    Same attributes as the BaseModelOutputWithPastAndCrossAttentions class (output of the regular
    encoder, T5Stack)

    And some attributes from the VAE model:
        recon_loss: reconstruction loss (identical to loss, if we don't use VAE)
        reg_loss: either KL(q(z|x)||p(z)) or MMD(q(z)||p(z))
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    vae_latent_s: Optional[torch.FloatTensor] = None
    recon_loss: Optional[torch.FloatTensor] = None
    reg_loss: Optional[torch.FloatTensor] = None


class VAEBase(ABC):
    beta: float  # to be specified by Trainer, before calling model forward
    reg_loss_type: str  # to be specified during init

    def compute_reg_loss(self, q_phi: Normal, z: torch.Tensor, prior: Optional[Normal]=None):
        if prior is None:
            # N(0,I) prior: same shape and device than q_phi
            prior = Normal(
                loc=torch.zeros_like(q_phi.loc),
                scale=torch.ones_like(q_phi.scale),
            )

        if self.reg_loss_type == "kl":
            kl_div = kl_divergence(q_phi, prior)
            # (N, T, dim_z) or (N, dim_z), make sure we do have vae_latent on the last dim
            #assert kl_div.shape[-1] == self.model_dim
            # reduce to a scalar by:
            #   - summing over latent dim, to obtain the D_KL between multivariate Normal distributions
            #   - taking the mean over batch and sequence dim, to match the
            #       CrossEntropyLoss (which takes mean over N and T as well)
            kl_div = kl_div.sum(dim=-1).mean()
            return kl_div
        elif self.reg_loss_type == "mmd":
            # https://github.com/amir-abdi/disentanglement-pytorch/blob/master/models/infovae.py
            z_prior = prior.rsample()
            mmd = self.compute_mmd(z, z_prior)
            return mmd
        else:
            raise ValueError

    # MMD-VAE specific methods
    @staticmethod
    @abstractmethod
    def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
        pass

    def compute_mmd(self, x: torch.Tensor, y: torch.Tensor):
        # original implementation: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        # MMD = E[k(x,x)] + E[k(y,y)] - 2 * E[k(x,y)]
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


class DT7Base(T5PreTrainedModel):
    # Based on T5ForConditionalGeneration, from transformers 4.9.1
    # https://huggingface.co/transformers/_modules/transformers/models/t5/modeling_t5.html#T5ForConditionalGeneration
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    # Encoder class, to define (e.g. T5Stack for regular T5)
    encoder_cls = None

    def __init__(
        self,
        config,
        specify_target_with_prefix: bool,
        generate_text_token_id: int,
        generate_data_token_id: int,
        encoder_kwargs: dict={},
    ):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = self.encoder_cls(encoder_config, self.shared, **encoder_kwargs)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.specify_target_with_prefix = specify_target_with_prefix
        if not specify_target_with_prefix:
            self.generate_text_token_id = generate_text_token_id
            self.generate_data_token_id = generate_data_token_id

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)
                    ),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past

    def _shift_right(self, input_ids, target):
        """
        Override the _shift_right method of T5PreTrainedModel, to add a
        custom token at the beginning of input_ids, (if self.specify_target_with_prefix).

        Instead of the default decoder_start_token_id (configured to pad_token_id for T5
        by default), use either a text_decoder_start_token_id or a data_decoder_start_token_id
        depending on the desired target (data or text generation).

        This is used during the forward, in training mode, to shift the labels and obtain decoder inputs
        During eval, we are using the base generate method, which takes care of using
        the right decoder_input_ids with the appropriate decoder_start_token_id argument.

        Args:
            input_ids:
            target: 'text' or 'data'

        Returns:

        """
        pad_token_id = self.config.pad_token_id
        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."

        if self.specify_target_with_prefix:
            # target is already specified as a prefix in the input sequence
            decoder_start_token_id = pad_token_id
        elif target == "text":
            decoder_start_token_id = self.generate_text_token_id
        elif target == "data":
            decoder_start_token_id = self.generate_data_token_id
        else:
            raise ValueError(f"Target (text/data) should be specified")
        assert (
            decoder_start_token_id is not None
        ), f"decoder_start_token_id (for target={target}) has not been defined)"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def generate_with_target(
        self,
        input_ids: torch.Tensor,
        target: str,
        prefixes: Dict[str, torch.Tensor],
        max_seq_length: int,
        method: str,
        num_beams=-1,
        **other_kwargs,
    ):
        """
        Call `generate` on our model, specifying the target format (data/text)

        Args:
            input_ids:
            target: 'text' or 'data'
            prefixes:
            max_seq_length:
            method: 'greedy', 'beam_search', 'sample' or 'top_k'
            num_beams: Used only when method='beam_search'
            other_kwargs: 'vae_latent' or 'source' format for instance. Will be passed to the model and the encoder

        Returns:

        """
        # specify the target format to the model
        if self.specify_target_with_prefix:
            if 'format_target' in other_kwargs:
                assert target == 'data'
                format_target = other_kwargs.pop('format_target')
                input_ids = add_target_prefix(
                    input_ids=input_ids,
                    target=format_target,
                    prefixes=prefixes
                )
            else:
                input_ids = add_target_prefix(
                    input_ids=input_ids,
                    target=target,
                    prefixes=prefixes
                )
            decoder_start_token_id = None
        else:
            # don't touch the encoder_input_ids, but tell the decoder the target format
            if target == "text":
                decoder_start_token_id = self.generate_text_token_id
            elif target == "data":
                decoder_start_token_id = self.generate_data_token_id
            else:
                raise ValueError

        kwargs = {
            "input_ids": input_ids,
            "max_length": max_seq_length,
            "decoder_start_token_id": decoder_start_token_id,
        }
        kwargs.update(other_kwargs)

        # generate text according to the specified decoding method
        if method == "greedy":
            # nothing to change to config
            pass
        elif method == "beam_search":
            assert num_beams > 1
            kwargs["num_beams"] = num_beams
            kwargs["early_stopping"] = True
        elif method == "sample":
            kwargs["do_sample"] = True
            kwargs["top_k"] = 0
        elif method == "top_k":
            kwargs["do_sample"] = True
            kwargs["top_k"] = 50
        else:
            raise ValueError

        self.eval()
        with torch.no_grad():
            prediction_ids = self.generate(**kwargs)
        return prediction_ids


class DT7NonVAE(DT7Base):
    encoder_cls = T5Stack

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        inputs = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
        return inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target=None,
    ):
        """
        target: Target format (can be "data" or "text"). Only used when we don't specify it
        already with an added prefix in the inputs, otherwise it can be None.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # make things easier to read by using ModelOutputs objects for encoder/decoder outputs
        # -> just make sure this is never overridden to False
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        assert return_dict
        # same for model_parallel (and make sure we don't use it)
        assert not self.model_parallel
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # to be fed to the decoder
        hidden_states = encoder_outputs.last_hidden_state

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(input_ids=labels, target=target)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs.last_hidden_state

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return DT7ModelOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class ImDisStyleVAET5Encoder(T5Stack):
    """
    Disentangle style and content for multi-format handling
    Encode the style from content AND input embedding
    """
    use_style_token: bool
    def __init__(
        self,
        config: T5Config,
        embed_tokens: nn.Embedding = None,
        **encoder_kwargs):
        super().__init__(config, embed_tokens)
        self.s_y_dim = encoder_kwargs['s_y_dim']
        self.s_x_dim = encoder_kwargs['s_x_dim']
        model_dim = config.d_model
        # additional layers for the variational posterior q(s_x|c,x)
        self.mu_s_x = nn.Linear(model_dim, self.s_x_dim)
        self.log_sigma_s_x = nn.Linear(model_dim, self.s_x_dim)
        self.s_x_embed = nn.Linear(self.s_x_dim, model_dim)
        # and for q(s_y|c,y)
        self.s_y = nn.Linear(model_dim, self.s_y_dim)
        self.s_y_embed = nn.Embedding(self.s_y_dim, model_dim)

    def forward(self, *args, **kwargs):
        source_format = None
        vae_s = None
        q_phi_s = None

        if "target" in kwargs:
            kwargs.pop("target")
        if "source" in kwargs:
            source_format = kwargs.pop("source")
        if "vae_latent_s" in kwargs:
            vae_s = kwargs.pop("vae_latent_s")

        # make sure the output is always BaseModelOutputWithPastAndCrossAttentions
        assert kwargs["return_dict"]

        # Retrieve last hidden states
        encoder_outputs = super().forward(*args, **kwargs)
        
        if vae_s is None and source_format is not None:
            # (N, model_dim)
            if self.use_style_token:
                # representation of our special [STYLE] token, after encoding
                style_hidden_state = encoder_outputs.last_hidden_state[:, 0, :].clone()
            else:
                # simply the mean of encoder outputs sequence
                style_hidden_state = encoder_outputs.last_hidden_state.mean(dim=1)

            if source_format == "text":
                mu_s = self.mu_s_x(style_hidden_state)
                log_sigma_s = self.log_sigma_s_x(style_hidden_state)
                # Sample for style
                q_phi_s = Normal(loc=mu_s, scale=torch.exp(log_sigma_s))
                vae_s = q_phi_s.rsample()  # dimension (N, style_dim)
            
            return VariationalT5EncoderOutput(
                q_phi_s=q_phi_s,
                vae_latent_s=vae_s,
            )

        return VariationalT5EncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            q_phi_s=q_phi_s,
            vae_latent_s=vae_s,
        )


class DT7ImDisStyleVAE(VAEBase, DT7Base):
    encoder_cls = ImDisStyleVAET5Encoder

    def __init__(
        self,
        config,
        specify_target_with_prefix: bool,
        generate_text_token_id: int,
        generate_data_token_id: int,
        use_style_token: bool,
        nb_formats: int,
        s_x_dim: int,
        reg_loss: Optional[str] = None,
    ):
        encoder_kwargs = {
            's_y_dim': nb_formats,
            's_x_dim': s_x_dim
        }
        DT7Base.__init__(
            self,
            config,
            specify_target_with_prefix=specify_target_with_prefix,
            generate_text_token_id=generate_text_token_id,
            generate_data_token_id=generate_data_token_id,
            encoder_kwargs=encoder_kwargs
        )
        self.use_style_token = use_style_token
        self.encoder.use_style_token = use_style_token
        self.reg_loss_type = reg_loss

    @staticmethod
    def compute_kernel(x: torch.Tensor, y: torch.Tensor):
        """
        Compute the k(x,y) values with latent variable samples x,y. All combinations of x and y
        across batch dim are considered, but not across sequence dim, for memory reasons.

        Compute the kernel depending on the size of x
        """
        if len(x.shape) == 2:
            # style latent variables
            N, dim = x.shape
            tiled_x = x.view(N, 1, dim).repeat(1, N, 1)
            tiled_y = y.view(1, N, dim).repeat(N, 1, 1)
        elif len(x.shape) == 3:
            # content latent variable
            N, T, dim = x.shape
            tiled_x = x.view(N, 1, T, dim).repeat(1, N, 1, 1)
            tiled_y = y.view(1, N, T, dim).repeat(N, 1, 1, 1)
        else:
            raise ValueError

        # compute RBF kernel k(x,y)
        sigma_sqr = dim ** 2 / 2
        squared_dist_xy = torch.sum((tiled_x - tiled_y) ** 2, dim=-1)
        return torch.exp(-0.5 * squared_dist_xy / sigma_sqr)
    
    def encode_content(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return encoder_outputs.last_hidden_state

    def encode_and_compute_reg_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source: str,
    ):
        """
        Only do the style encoding part of the forward
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            source=source,
            return_dict=True,
        )
        reg_style_loss = self.compute_reg_loss(
            q_phi=encoder_outputs.q_phi_s,
            z=encoder_outputs.vae_latent_s,
        )

        return DT7ModelOutput(reg_loss=reg_style_loss,
                              vae_latent_s=encoder_outputs.vae_latent_s)
    
    def generate_with_target(
        self,
        input_ids: torch.Tensor,
        target: str,
        prefixes: Dict[str, torch.Tensor],
        max_seq_length: int,
        method: str,
        num_beams=-1,
        **other_kwargs,
    ):
        """
        Call `generate` on our model, specifying the target format (data/text)

        Args:
            input_ids:
            target: 'text' or 'data'
            prefixes:
            max_seq_length:
            method: 'greedy', 'beam_search', 'sample' or 'top_k'
            num_beams: Used only when method='beam_search'
            other_kwargs: 'vae_latent' or 'source' format for instance. Will be passed to the model and the encoder

        Returns:

        """
        # specify the target format to the model
        if self.specify_target_with_prefix:
            input_ids = add_target_prefix(
                input_ids=input_ids,
                target=target,
                prefixes=prefixes
            )
            decoder_start_token_id = None
        else:
            # don't touch the encoder_input_ids, but tell the decoder the target format
            if target == "text":
                decoder_start_token_id = self.generate_text_token_id
            elif target == "data":
                decoder_start_token_id = self.generate_data_token_id
            else:
                raise ValueError

        kwargs = {
            "input_ids": input_ids,
            "max_length": max_seq_length,
            "decoder_start_token_id": decoder_start_token_id,
            "target": target
        }
        kwargs.update(other_kwargs)

        # generate text according to the specified decoding method
        if method == "greedy":
            # nothing to change to config
            pass
        elif method == "beam_search":
            assert num_beams > 1
            kwargs["num_beams"] = num_beams
            kwargs["early_stopping"] = True
        elif method == "sample":
            kwargs["do_sample"] = True
            kwargs["top_k"] = 0
        elif method == "top_k":
            kwargs["do_sample"] = True
            kwargs["top_k"] = 50
        else:
            raise ValueError

        self.eval()
        with torch.no_grad():
            prediction_ids = self.generate(**kwargs)
        return prediction_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        if "vae_latent_s" in kwargs:
            # use vae_latent specified as a kwarg to the generate method
            pass
        else:
            if kwargs["target"] == 'text':
                #prior = Normal(
                #    loc=torch.zeros((encoder_outputs.last_hidden_state.size(0), self.encoder.s_x_dim)).to(input_ids.device),
                #    scale=torch.ones((encoder_outputs.last_hidden_state.size(0), self.encoder.s_x_dim)).to(input_ids.device),
                #)
                #encoder_outputs.vae_latent_s = prior.rsample()
                encoder_outputs.vae_latent_s = torch.zeros((encoder_outputs.last_hidden_state.size(0), self.encoder.s_x_dim)).to(input_ids.device)
            elif kwargs["target"] == 'data':
                # randomly sample a format (preferably a target format)
                prior = Categorical(torch.full((encoder_outputs.last_hidden_state.size(0), self.encoder.s_y_dim), 1/self.encoder.s_y_dim))
                emb_id = prior.sample()
                encoder_outputs.vae_latent_s = torch.nn.functional.one_hot(emb_id, num_classes=self.encoder.s_y_dim).to(input_ids.device)

        inputs = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "target": kwargs["target"]
        }
        return inputs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: VariationalT5EncoderOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """
        Expand tensors across the batch dimension, e.g. to have num_beams*batch_size
        instead of batch_size. Called during beam search generation for instance.

        We redefine this function to also expand vae_mu_z/vae_sigma_z (in encoder_outputs).
        """
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )

        if is_encoder_decoder:

            assert encoder_outputs is not None
            encoder_outputs[
                "last_hidden_state"
            ] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )

            # VAE specific code
            if encoder_outputs.vae_latent_s is not None:
                encoder_outputs["vae_latent_s"] = encoder_outputs.vae_latent_s.index_select(
                    0, expanded_return_idx.to(encoder_outputs.vae_latent_s.device)
                )
            
            if encoder_outputs.q_phi_s is not None:
                loc_s = encoder_outputs.q_phi_s.loc.index_select(
                    0, expanded_return_idx.to(encoder_outputs.q_phi_s.loc.device)
                )
                scale_s = encoder_outputs.q_phi_s.scale.index_select(
                    0, expanded_return_idx.to(encoder_outputs.q_phi_s.scale.device)
                )
                encoder_outputs["q_phi_s"] = Normal(loc_s, scale_s)

            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs
    
    def get_latent_encoding(self, encoder_outputs, target):
        # Concatenate content and style latent codes and embed into latent code
        latent_encoding = encoder_outputs.last_hidden_state.clone()
        vae_latent_style = encoder_outputs.vae_latent_s.clone()

        if target == 'data':
            weighted_s_y = self.encoder.s_y_embed.weight.clone().unsqueeze(0).expand(vae_latent_style.size(0), vae_latent_style.size(1), -1)*vae_latent_style.unsqueeze(-1)
            vae_latent_style = weighted_s_y.sum(1)
        elif target == 'text':
            vae_latent_style = self.encoder.s_x_embed(vae_latent_style)
        
        if self.use_style_token:
            latent_encoding[:, 0] = vae_latent_style
        else:
            # simply add both latent codes
            expanded_vae_s = vae_latent_style.unsqueeze(1).expand_as(latent_encoding)
            latent_encoding = latent_encoding + expanded_vae_s
        
        return latent_encoding

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target=None,
        source=None,
    ):
        """
        target: Target format (can be "data" or "text"). Only used when we don't specify it
        already with an added prefix in the inputs, otherwise it can be None.
        """
        if input_ids is not None and labels is None and encoder_outputs is None:
            return self.encode_and_compute_reg_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                source=source,
            )
            

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # make things easier to read by using ModelOutputs objects for encoder/decoder outputs
        # -> just make sure this is never overridden to False
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        assert return_dict
        # same for model_parallel (and make sure we don't use it)
        assert not self.model_parallel
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                source=source,
            )
        elif list(encoder_outputs.keys()) == ["vae_latent_s"]:
            # Encode with a given s
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                source=source,
                vae_latent_s=encoder_outputs.vae_latent_s,
            )

        # to be fed to the decoder
        hidden_states = self.get_latent_encoding(encoder_outputs, target)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(input_ids=labels,
                                                  target=target)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs.last_hidden_state

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss, recon_loss = None, None

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            recon_loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )
            # loss = -L_elbo = -log p(x|z) 
            loss = recon_loss

        return DT7ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            vae_latent_s=encoder_outputs.vae_latent_s,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

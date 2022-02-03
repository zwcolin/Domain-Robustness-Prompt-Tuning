import os
from pathlib import Path

from transformers import T5ForConditionalGeneration, GPT2PreTrainedModel
import torch
import torch.nn as nn


class T5PromptTuningMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        n_tokens: int = None,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        return model

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = (
                self.get_input_embeddings().weight[:n_tokens].clone().detach()
            )
        #             init_prompt_value = self.transformer.wte.weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        self.soft_prompt = nn.Embedding(n_tokens, self.config.d_model)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        #         inputs_embeds = self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        to_encoder_only=False,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        # for training, extend the attention mask to include input embeddings, but not for inference,
        # where greedy search only requires encoder outputs and decoder_input ids and the shape needs to match
        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = self._extend_attention_mask(
                    decoder_attention_mask
                ).to(self.device)

        if to_encoder_only:
            return self.encoder(inputs_embeds=inputs_embeds, return_dict=True)

        # for inference (i.e. generate) - build pipeline for generate function
        if decoder_input_ids is not None:
            return super().forward(
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                use_cache=use_cache,
                return_dict=return_dict,
            )

        # for training
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
            encoder_outputs=encoder_outputs,
        )


class T5PromptTuningLM(T5PromptTuningMixin, T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)


class PrefixTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""

    def __init__(
        self,
        config,
        model_gpt2,
        optim_prefix=False,
        preseqlen=5,
        use_infix=False,
        deep_param=False,
    ):
        super().__init__(config)
        print("under the PrefixTuning model")

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        if hasattr(config, "optim_prefix"):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, "preseqlen") and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, "use_infix"):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, "_my_arg_tune_mode"):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = "prefixtune"

        if hasattr(config, "_my_arg_task_mode"):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = "underspecified"
            assert False, "the task is underspecified"

        if hasattr(config, "train_weights"):
            self.train_weights = config.train_weights == "yes"
        else:
            assert False, "unspecified train weights"

        if hasattr(config, "format_mode"):
            self.format_mode = config.format_mode
        else:
            self.format_mode = "cat"

        if hasattr(config, "prefix_dropout"):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        if hasattr(config, "init_random"):
            self.init_random = config.init_random == "yes"
        else:
            self.init_random = False

        if hasattr(config, "mid_dim"):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if True:
            self.mode_para = 0
            print("PrefixTuning")
            print(
                "preseqlen is {}, optimizing the prefix directly".format(self.preseqlen)
            )

            if not deep_param:
                print("[Full prefix-tuning Setting :) ]")
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
                )
                if self.use_infix:
                    self.wte2 = nn.Embedding(self.preseqlen, config.n_embd)
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5
            else:
                print("[DOUBLE CHECK]: DEEP MLP")
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.forward = self.forward_infix

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print("total param is {}".format(total_param))

    def get_gold_init(self, gpt2, sample_input):
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            output = gpt2(
                sample_input.to(gpt2.device), return_dict=True, use_cache=True
            )
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)
        return output

    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        temp_control = self.control_trans.view(
            1,
            self.preseqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        ).expand(bsz, -1, -1, -1, -1)
        temp_control = self.dropout(temp_control)
        past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p5_infix(
        self, src, control_code=None, gpt2=None, bsz=None, attn_mask=None
    ):
        # VERSION1. infixing by taking in the last layer of the hidden states as input.

        # VERSION2. infixing by pretending some input to first get the history, then add upon them.
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])

        temp_emb = self.wte2(input_tokens)
        src_emb = gpt2.transformer.wte(src)
        total_emb = torch.cat([src_emb, temp_emb], dim=1)  # bsz, seqlen, dim
        src_out = gpt2(
            inputs_embeds=total_emb,
            attention_mask=attn_mask,
            use_cache=True,
            return_dict=True,
        )
        src_past_key_vals = src_out.past_key_values
        src_past_key_vals = torch.cat(src_past_key_vals, dim=0)
        # print(src_past_key_vals.shape, past_key_values.shape) # the src should be longer than past.
        # get a zero mask.
        _, src_len = src.shape
        nl, nb, nh, _, ndim = past_key_values.shape
        zero_mask = torch.zeros(nl, nb, nh, src_len, ndim).to(self.device)
        # print(zero_mask.shape, past_key_values.shape)
        past_key_values = torch.cat([zero_mask, past_key_values], dim=3)
        # print(past_key_values.shape)
        past_key_values = past_key_values + src_past_key_vals

        # add them together.
        past_key_values = past_key_values.split(2)

        return past_key_values

    def forward(
        self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
    ):

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        else:
            past_key_values_prompt = self.get_prompt(
                control_code, gpt2=gpt2_model, bsz=bsz
            )
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(
            input_ids=input_ids,
            control_code=None,
            weights=weights,
            emb_match=emb_match,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return output

    def forward_infix(
        self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
    ):

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(
                src, None, gpt2=gpt2_model, bsz=bsz
            )
            attention_mask = torch.cat(
                [src_attn, src_attn, tgt_attn], dim=1
            )  # bsz, seqlen
        else:
            infix_attn = torch.ones(bsz, self.preseqlen).bool().to(self.device)
            attention_mask = torch.cat(
                [src_attn, infix_attn, tgt_attn], dim=1
            )  # bsz, seqlen
            partial_attn_mask = torch.cat([src_attn, infix_attn], dim=1)  # bsz, seqlen
            past_key_values_prompt = self.get_prompt(
                src, None, gpt2=gpt2_model, bsz=bsz, attn_mask=partial_attn_mask
            )
            # print(src_attn)
            # print()
            # print(infix_attn)
            # infix_attn = torch.ones(bsz, self.preseqlen).to(self.device)
            # attention_mask = torch.cat([src_attn, infix_attn, tgt_attn], dim=1)  # bsz, seqlen

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        output = gpt2_model(
            input_ids=input_ids,
            control_code=None,
            weights=weights,
            emb_match=emb_match,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return output

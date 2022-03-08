from transformers import GPT2PreTrainedModel, T5PreTrainedModel, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import torch.nn as nn


class PrefixTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""

    def __init__(
        self,
        config,
        model_gpt2,
        optim_prefix=False,
        preseqlen=5,
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

        self.format_mode = "cat"

        self.prefix_dropout = 0.0

        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        self.init_random = False

        self.mid_dim = 512

        if True:
            self.mode_para = 0
            print("PrefixTuning")
            print(
                "preseqlen is {}, optimizing the prefix directly".format(self.preseqlen)
            )

            print("[Full prefix-tuning Setting :) ]")
            self.input_tokens = torch.arange(self.preseqlen).long()
            self.wte = nn.Embedding(self.preseqlen, config.n_embd)
            self.control_trans = nn.Sequential(
                nn.Linear(config.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
            )

            self.get_prompt = self.get_prompt_p5

        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            total_param += param.numel()
        print("total param is {}".format(total_param))

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


class T5PrefixTuning(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.preseqlen = config.preseqlen
        self.mid_dim = 512
        self.prefix_dropout = 0.0

        model_name = config.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=f"cache/{model_name}-s3"
            )
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=f"cache/{model_name}-s3"
        )
        self.config = self.pretrain_model.config
        self.match_n_layer = self.config.num_layers
        self.match_n_head = self.config.num_heads
        self.n_embd = self.config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
        )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
        )

        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
        )

        self.dropout = nn.Dropout(self.prefix_dropout)
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

    def get_prompt(self, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result


    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None,
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(bsz=bsz)

        output = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
        )
        return output

    def generate(self,
                 input_ids,
                 attention_mask=None,
                 **kwargs):

        bsz = input_ids.shape[0]
        past_prompt = self.get_prompt(bsz=bsz)
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
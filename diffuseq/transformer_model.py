from transformers import AutoConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch

import gc
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding
)

class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
        sc_rate=0,
        predict_t=False
    ):
        super().__init__()

        if config is None:
            if config_name == "bert-base-uncased":
                config = AutoConfig.from_pretrained(config_name)
            elif config_name == "bert-de2en":
                config = AutoConfig.from_pretrained("bert-base-uncased")
                config.num_attention_heads=8
                config.hidden_size=512
                config.intermediate_size=1024
            elif init_pretrained == 'qwen':
                # Derive BertConfig from Qwen3's architecture parameters so that
                # the diffusion backbone mirrors Qwen3's capacity while remaining
                # compatible with BertEncoder.
                from transformers import BertConfig
                _qwen_cfg = AutoConfig.from_pretrained(config_name, trust_remote_code=True)
                config = BertConfig(
                    hidden_size=_qwen_cfg.hidden_size,
                    intermediate_size=_qwen_cfg.intermediate_size,
                    num_hidden_layers=_qwen_cfg.num_hidden_layers,
                    num_attention_heads=_qwen_cfg.num_attention_heads,
                    max_position_embeddings=_qwen_cfg.max_position_embeddings,
                )
            config.hidden_dropout_prob = dropout
                
        

        self.input_dims = input_dims
        if sc_rate > 0:
            self.input_dims = input_dims * 2
        self.predict_t = predict_t
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, input_dims)
        self.lm_head = nn.Linear(input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(self.input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            # self.lm_head.weight.requires_grad = False
            # self.word_embedding.weight.requires_grad = False
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            print("BertConfig: ", config)
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        elif init_pretrained == 'qwen':
            from transformers import AutoModelForCausalLM
            print('initializing word embedding from pretrained Qwen3...')
            print(config)

            # Load the Qwen3 model in fp16 on CPU to save peak memory.
            # Only the embedding table is kept; the rest is deleted afterwards.
            qwen_model = AutoModelForCausalLM.from_pretrained(
                config_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map='cpu',
            )

            qwen_emb_weight = qwen_model.model.embed_tokens.weight.detach().float()
            qwen_emb_dim = qwen_emb_weight.shape[1]  # e.g. 5120 for Qwen3-32B

            with th.no_grad():
                if qwen_emb_dim == input_dims:
                    # Dimensions already match; copy directly.
                    self.word_embedding.weight.copy_(qwen_emb_weight)
                else:
                    # Project qwen_emb_dim → input_dims with a scaled normal projection.
                    proj = nn.Linear(qwen_emb_dim, input_dims, bias=False)
                    nn.init.normal_(proj.weight, std=1.0 / (qwen_emb_dim ** 0.5))
                    self.word_embedding.weight.copy_(proj(qwen_emb_weight))

            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

            # Free Qwen3 weights; only the projected embedding is retained.
            del qwen_model, qwen_emb_weight
            gc.collect()

            # Diffusion backbone: BertEncoder shaped by the Qwen3-derived BertConfig.
            self.input_transformers = BertEncoder(config)
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(timesteps)
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h
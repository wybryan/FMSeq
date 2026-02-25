# Discrete Token Loss Explained

## Overview

FMSeq's training loss (Eq. 7 in the paper) combines two terms:

```
L_E2E = L_FM (flow matching) + L_R (rounding / discrete token loss)
```

In code (`gaussian_diffusion.py:720`):
```python
terms["loss"] = terms["mse"] + decoder_nll
```

This document focuses on `decoder_nll` — the discrete token loss (rounding loss).

---

## What is `decoder_nll`?

**Paper reference**: L_R = −E[ log p_φ(w^z | z_0) ] (Eq. 6)

**Code** (`gaussian_diffusion.py:715`):
```python
decoder_nll = self._token_discrete_loss(x_start, get_logits, input_ids_x)
```

### Purpose

Flow matching loss backpropagates gradients to both the embedding matrix and the model parameters. This can cause the embedding matrix to **collapse** — when all word vectors converge to the same point, the data distribution becomes a Dirac distribution that is trivially easy to model. The rounding loss regularizes the embedding by ensuring each continuous embedding can be projected back to its original discrete token.

---

## Inputs

### `x_start` — the clean embedding (y_0 in paper)

Constructed at `gaussian_diffusion.py:632-640`:
```python
x_start_mean = model.model.module.get_embeds(input_ids_x)  # embed tokens
std = th.zeros_like(...)  # no noise
x_start = self._get_x_start(x_start_mean, std)  # = x_start_mean + 0
```

`x_start` is simply `EMB(w^z)` — the embedding matrix applied to the concatenated token IDs (w^x ⊕ w^y), with no noise added.

### `get_logits` — the rounding function

Assigned at `gaussian_diffusion.py:677`:
```python
get_logits = model.model.module.get_logits
```

With the default `logits_mode=1` (`transformer_model.py:42`), this is:
```python
return self.lm_head(hidden_repr)  # nn.Linear(d, vocab_size)
```

Critically, `lm_head.weight` is **tied** to `word_embedding.weight` (`transformer_model.py:106`):
```python
self.lm_head.weight = self.word_embedding.weight
```

So `get_logits(x)` computes `x @ E^T` — the dot product between the continuous representation and every token's embedding vector. The result is a logit score for each vocabulary token.

### `input_ids_x` — the ground truth token IDs

The original discrete token IDs of the concatenated sequence w^z = w^x ⊕ w^y. Shape: `[batch, seq_len]`.

---

## The Loss Computation

Inside `_token_discrete_loss` (`gaussian_diffusion.py:550-569`):

```python
def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):
    logits = get_logits(x_t)                    # [batch, seq_len, vocab_size]
    loss_fct = th.nn.CrossEntropyLoss(reduction='none')
    decoder_nll = loss_fct(
        logits.view(-1, logits.size(-1)),        # [batch*seq_len, vocab_size]
        input_ids.view(-1)                       # [batch*seq_len]
    ).view(input_ids.shape)                      # [batch, seq_len]
    # no mask passed for decoder_nll, so average over full sequence:
    decoder_nll = decoder_nll.mean(dim=-1)       # [batch]
```

### Step by step

1. **Project to vocab space**: `logits = x_start @ E^T` → `[batch, seq_len, vocab_size]`
2. **Cross-entropy**: For each position, compute `-log(softmax(logits)[correct_token_id])`
3. **Average over sequence**: Since no mask is passed, averages over all positions (both source and target tokens)

This is effectively cross-entropy against one-hot labels — PyTorch's `CrossEntropyLoss` with integer targets is mathematically equivalent to cross-entropy against one-hot vectors.

---

## The Other `_token_discrete_loss` Call

There is a second call at `gaussian_diffusion.py:717`:

```python
terms["nll"] = self._token_discrete_loss(
    model_out_x_start, get_logits, input_ids_x, mask=input_ids_mask, truncate=True, t=t
)
```

| | `decoder_nll` (L_R) | `terms["nll"]` |
|---|---|---|
| **Input embedding** | `x_start` (clean ground truth embedding) | `model_out_x_start` (model's predicted x_0) |
| **Mask** | None (all tokens) | `input_ids_mask` (target tokens only) |
| **In training loss?** | Yes | No (diagnostic/monitoring only) |
| **Purpose** | Embedding regularization | Measures rounding quality of model predictions |

`model_out_x_start` is recovered from the velocity prediction (`gaussian_diffusion.py:701-703`):
```python
# x_t = x_start + t * (noise - x_start)   →   x_0 = x_t - t * velocity
model_out_x_start = x_t - rescale_t * model_output
```

---

## Summary

The discrete token loss (`decoder_nll`) is a round-trip consistency check:

```
tokens → EMB(tokens) → EMB(tokens) @ EMB^T → softmax → cross_entropy(tokens)
```

It ensures the embedding space stays well-structured so that discrete tokens can always be recovered from their continuous representations by nearest-neighbor lookup. Without it, the embedding matrix collapses and the model becomes unusable.

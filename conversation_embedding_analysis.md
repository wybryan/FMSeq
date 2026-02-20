# FMSeq Embedding 机制分析

## Q1: src 和 trg 分别是如何做 embedding 的？哪些是 random init，哪些是 pretrained？

### 核心结论：src 和 trg 共享同一个 embedding 层

没有独立的 src embedding 和 trg embedding，两者都通过同一个 `word_embedding` 处理，区别通过 `input_mask` 实现。

---

### 一、两套 Embedding 对象及其来源

#### 对象 1: `model_weight` — 数据预处理用 embedding

`basic_utils.py:180-202`

```python
def load_model_emb(args, tokenizer):
    model = torch.nn.Embedding(tokenizer.vocab_size, args.hidden_dim)
    # ...
    torch.nn.init.normal_(model.weight)   # 始终是 random normal init
    torch.save(model.state_dict(), path_save)
```

**始终 random init**，无论 `init_pretrained` 是什么。

---

#### 对象 2: `model.word_embedding` — 模型内部 embedding（训练真正使用的）

`diffuseq/transformer_model.py:82-163`

初始化分三条路径，由 `init_pretrained` 参数决定：

**路径 A: `init_pretrained='no'`（random）**

```python
# Line 82-85
self.word_embedding = nn.Embedding(vocab_size, input_dims)  # PyTorch 默认 normal(0,1)
self.lm_head = nn.Linear(input_dims, vocab_size)
with th.no_grad():
    self.lm_head.weight = self.word_embedding.weight  # 权重绑定

# Line 117-123
elif init_pretrained == 'no':
    self.input_transformers = BertEncoder(config)
    self.position_embeddings = nn.Embedding(...)  # 也是 random
    self.LayerNorm = nn.LayerNorm(...)
```

**路径 B: `init_pretrained='bert'`（pretrained）**

```python
# Line 98-115
elif init_pretrained == 'bert':
    temp_bert = BertModel.from_pretrained(config_name, config=config)
    self.word_embedding = temp_bert.embeddings.word_embeddings  # ← 直接用 BERT 权重
    with th.no_grad():
        self.lm_head.weight = self.word_embedding.weight
    self.input_transformers = temp_bert.encoder                 # ← BERT encoder 权重
    self.position_embeddings = temp_bert.embeddings.position_embeddings  # ← BERT 位置编码
    self.LayerNorm = temp_bert.embeddings.LayerNorm             # ← BERT LayerNorm
```

**路径 C: `init_pretrained='qwen'`（pretrained + 可选投影）**

```python
# Line 125-163
elif init_pretrained == 'qwen':
    qwen_model = AutoModelForCausalLM.from_pretrained(config_name, ...)
    qwen_emb_weight = qwen_model.model.embed_tokens.weight.detach().float()
    qwen_emb_dim = qwen_emb_weight.shape[1]
    with th.no_grad():
        if qwen_emb_dim == input_dims:
            self.word_embedding.weight.copy_(qwen_emb_weight)   # 直接复制
        else:
            proj = nn.Linear(qwen_emb_dim, input_dims, bias=False)
            nn.init.normal_(proj.weight, std=1.0 / (qwen_emb_dim ** 0.5))  # 投影层 random
            self.word_embedding.weight.copy_(proj(qwen_emb_weight))  # 投影后复制
    # Transformer backbone 仍然是 random BertEncoder（非 Qwen 的 transformer）
    self.input_transformers = BertEncoder(config)
```

---

### 二、关键证明：`model_weight` 在训练中被覆盖

`diffuseq/gaussian_diffusion.py:625-638`

```python
def training_losses_seq2seq(self, model, x_start, t, ...):
    # x_start 来自 TextDataset，使用 model_weight（random init）
    # 注释说明：# not used unless fixing the input embeddings
    x_start_fix = x_start   # 保存但几乎不用

    input_ids_x = model_kwargs.pop('input_ids').to(t.device)

    # 真正用于训练的 embedding：来自 model.word_embedding
    x_start_mean = model.model.module.get_embeds(input_ids_x)  # Line 630

    std = _extract_into_tensor(...)
    std = th.zeros_like(std)   # std=0，即无噪声
    x_start = self._get_x_start(x_start_mean, std)  # ← x_start 被 x_start_mean 覆盖！
```

`std=0` 使 `_get_x_start` 退化为直接返回 `x_start_mean`，**数据集返回的 `x_start`（来自 `model_weight`）被完全丢弃**，训练实际使用 `model.word_embedding` 的输出。

---

### 三、src 和 trg 的区分方式

`diffuseq/text_datasets.py:104-130` 构建序列结构：

```
input_ids:  [src_tokens...] + [SEP] + [trg_tokens...]
input_mask: [0, 0, ..., 0,    0   ,  1, 1, ..., 1   ]
             ↑ src: mask=0，保持原始 embedding，不参与 loss
                                    ↑ trg: mask=1，参与扩散与 loss
```

---

### 四、总结表

| 对象 | 初始化方式 | 实际用途 |
|---|---|---|
| `model_weight`（`load_model_emb`） | **始终** random normal | 填充 DataLoader 的 tensor，训练中被覆盖 |
| `model.word_embedding`（`init_pretrained='no'`） | **random**（PyTorch `nn.Embedding` 默认） | 训练真正使用的 embedding，src+trg 共享 |
| `model.word_embedding`（`init_pretrained='bert'`） | **pretrained** BERT `word_embeddings` | 同上，+position、LayerNorm、encoder 都来自 BERT |
| `model.word_embedding`（`init_pretrained='qwen'`） | **pretrained** Qwen `embed_tokens`（可能经线性投影） | 同上；但 transformer backbone 仍是 random BertEncoder |
| `lm_head.weight` | **weight-tied** 到 `word_embedding.weight` | 解码 embedding → logits，共享权重 |

---

## Q2: trg 的 embedding 是 pretrained 还是 random？

**trg 和 src 的 embedding 初始化完全一致**，因为它们共享同一个 `word_embedding` 层。

取决于 `init_pretrained` 参数：

| `init_pretrained` | src embedding | trg embedding |
|---|---|---|
| `'no'` | random | random（相同权重）|
| `'bert'` | pretrained BERT | pretrained BERT（相同权重）|
| `'qwen'` | pretrained Qwen（投影后）| pretrained Qwen（投影后，相同权重）|

`diffuseq/transformer_model.py:82` 只定义了一个 embedding 层：

```python
self.word_embedding = nn.Embedding(vocab_size, input_dims)  # 唯一的 embedding
```

lookup 时 src 和 trg 的 token ID 都通过同一个函数：

```python
def get_embeds(self, input_ids):
    return self.word_embedding(input_ids)  # src+trg 拼接后一起查表
```

---

## Q3: trg 在哪里加噪声？

在 `diffuseq/gaussian_diffusion.py:643-650`，分两步：

**第一步：对整个序列做 flow matching 插值（src+trg 都加噪）**

```python
rescale_t = th.broadcast_to(t.float() * (1.0 / self.num_timesteps), ...)  # t 归一化到 [0,1]

x_t = x_start + (noise - x_start) * rescale_t  # flow matching: x_t = x0 + (x1-x0)*t
```

这是从 `x_start`（干净 embedding）到纯噪声 `noise` 的线性插值。

**第二步：把 src 部分强制还原为干净 embedding（只保留 trg 的噪声）**

```python
input_ids_mask_reshape = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape)
x_t = th.where(input_ids_mask_reshape == 0, x_start, x_t)
#                       ↑ src: mask=0，用干净的 x_start 覆盖
#                                              ↑ trg: mask=1，保留加噪后的 x_t
```

最终效果：

```
x_t = [ src_clean_emb... | SEP_clean | trg_noised_emb... ]
```

**模型的任务**就是从这个带噪的 `x_t` 中恢复出 trg 的干净 embedding `x_start`，src 全程作为条件保持不变。

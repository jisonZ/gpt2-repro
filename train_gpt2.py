from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import code
import time
import inspect
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a "bias", more of a mask
        # tril() returns a lower triangular matrix
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # split along the last dimension, each of size n_embd
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)

        # att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(C // self.n_head))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, n_head, T, head_dim)

        # NOTE: use flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, n_embd)
        return self.c_proj(y) # (B, T, n_embd)
    

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # NOTE: layernorm -> attention -> residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    # vocab_size: int = 50257
    # NOTE: Use exponential size for the vocab size
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # weights of the token embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # weights of the positional embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # the transformer layers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing between the token embeddings and the lm head
        # NOTE: tokens with similar semantic meaning should output similar probabilities from the lm head
        self.transformer.wte.weight = self.lm_head.weight

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initial params
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # NOTE: residule connection scale standard deviation
            # x = torch.zeros(768)
            # n = 100 # 100 layers
            # for i in range(n):
            #     x += torch.randn(768) # torch.randn has a standard deviation of 1
            # # x.std() = 10....
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emd = self.transformer.wpe(pos) # (T, n_embd)
        tok_emd = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emd + pos_emd
        # forward the transformer layers
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm (because we not have layer norm ahead of the feedforward)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # compute the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained model: %s" % model_type)

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768), # 124M parameters
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 350M parameters
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parameters
        }[model_type]
        config_args["vocab_size"] = 50257 # always 50257 for GPT model
        config_args["block_size"] = 1024 # always 1024 for GPT model

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # 

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # separate parameters that need decay and those that don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        # create adamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# --------------------------------------------------
# Simple Launch
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group

# set up DDP
# torch run command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # true if DDP is enabled
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    # current process rank (GPU0 will have rank 0, GPU1 will have rank 1, etc.)
    ddp_rank = int(os.environ['RANK'])
    # local rank is used in multi-node training setting
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # total number of processes
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # identify the master process to have ddp rank of 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# --------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, process_rank, world_size):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} steps")

        # state
        # NOTE: adjust current position based on the process rank
        self.current_position = 0 * self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T).to(device)
        y = buf[1:].view(B, T).to(device)

        # NOTE: adjust the current position based on the world size
        self.current_position += B*T * self.world_size

        # reset the position if we reach the end of the tokens
        if self.current_position + (B*T*self.world_size + 1) > len(self.tokens):
            self.current_position = 0 * self.B * self.T * self.process_rank

        return x, y

# --------------------------------------------------
# Use stochastic gradient to simulate mini-batch training
# GPT-2 has batch size of 0.5M
total_batch_size = 524288 # 2**19
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


# get data loader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, world_size=ddp_world_size)

# set precision
torch.set_float32_matmul_precision('high')

# get model
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

# NOTE: 
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# add weight decay
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1. linear warmup for warmup_steps, avoid zero learning rate
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2. if it > lr_decay_steps, use min_lr
    if it > max_steps:
        return min_lr
    # 3. between use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# NOTE: When uninitialized, the loss should be roughly -log(1/vocab_size)
# logits, loss = model(x, y)

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

for step in range(max_steps):
    t0 = time.time()

    optimizer.zero_grad()
    loss_accum = torch.tensor(0.0, device=device)

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        # NOTE: tensor.to(device) does not modify x in place, it returns a new tensor
        x, y = x.to(device), y.to(device)
        # code.interact(local=dict(globals(), **locals()))

        # use bfloat16 instead of float16
        # float16 range: ~6e-8 → 6e+4
        # bfloat16 range: ~1e-38 → 1e+38
        # NOTE: since bfloat16 has a larger range, it can represent smaller gradients, avoid gradient scalings.
        # NOTE: MIXED PRECISION! the params are still in float32, only some operations can autocast to bfloat16 (eg. matmul)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # NOTE: we need to normalize the loss by the number of gradient accumulation steps
        loss = loss / grad_accum_steps
        loss_accum += loss.item()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    
    # NOTE: reduce the loss from all processes
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        # NOTE: .item() behind the scene is convert the loss to a scalar and ship back to the CPU
        print(f"Step {step} | loss {loss_accum} | lr {lr:.2e} | norm {norm:.2f} | {dt*1000:.2f}ms | {tokens_per_sec:.2f} tokens/sec")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

# --------------------------------------------------
# num_return_sequences = 5
# max_length = 30

# # prefix tokens
# import tiktoken
# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode("Hello, I'm a language model")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# x = tokens.to('cuda')

# # model = GPT.from_pretrained("gpt2")
# model = GPT(GPTConfig())
# model.eval()
# model.to('cuda')

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(x) # (B, T, vocab_size)
#         # take the logits at the last position
#         logits = logits[:, -1, :] # (B, vocab_size)
#         # get the probabailities
#         probs = F.softmax(logits, dim=-1)
#         # do a top-k sampling
#         topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
#         # select a token from the top k probabilities
#         ix = torch.multinomial(topk_probs, num_samples=1)
#         # gather the indices
#         xIdx = torch.gather(topk_indices, dim=-1, index=ix)
#         # append the new tokens to the input
#         x = torch.cat((x, xIdx), dim=1)

# # print the generated text
# for i in range(num_return_sequences):
#     token = x[i, :max_length].tolist()
#     decoded = enc.decode(token)
#     print(">", decoded)
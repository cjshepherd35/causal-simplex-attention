# WikiText version of kmoment_simplex_ffwd.py, based on karpathy/timedwikikarpathy.py.
# Using 3 alternating kmoment attention and feedforward layers (like a normal Transformer).

import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


sys.stdout.reconfigure(encoding="utf-8")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device is: ', device)

# parameters to tweak
max_iters = int(os.getenv('MAX_ITERS', '50001'))
eval_iters = int(os.getenv('EVAL_ITERS', '10'))
eval_interval = int(os.getenv('EVAL_INTERVAL', '5000'))
n_embed = int(os.getenv('N_EMBED', '512'))
block_size = int(os.getenv('BLOCK_SIZE', '64'))
batch_size = int(os.getenv('BATCH_SIZE', '16'))
learning_rate = float(os.getenv('LEARNING_RATE', '3e-4'))
n_head = int(os.getenv('N_HEAD', '8'))
n_layer = int(os.getenv('N_LAYER', '6'))
dropout = float(os.getenv('DROPOUT', '0.2'))
moment_top_k = int(os.getenv('MOMENT_TOP_K', os.getenv('SIMPLEX_TOP_K', '8')))

dataset_range = int(os.getenv('DATASET_RANGE', '80000'))
vocab_size = int(os.getenv('VOCAB_SIZE', '800'))
num_merges = vocab_size - 256

assert vocab_size >= 256, "vocab_size must include the 256 raw byte tokens"
assert n_embed % n_head == 0, "n_embed must divide evenly across n_head"
assert moment_top_k > 0, "moment_top_k must be positive"
assert n_layer > 0, "n_layer must be positive"


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256
        text_chunks = self.compiled_pattern.findall(text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        for i in range(num_merges):
            stats = Counter()
            for chunk_ids in ids:
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    stats[pair] += 1
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i + 1) % 100 == 0:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx}")

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):
        all_ids = []
        for chunk in self.compiled_pattern.findall(text):
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                stats = Counter(zip(chunk_ids, chunk_ids[1:]))
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                chunk_ids = self._merge(chunk_ids, pair, self.merges[pair])
            all_ids.extend(chunk_ids)
        return all_ids

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            part_bytes.append(self.vocab[idx])
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")


def nearby_cache_path(filename):
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / filename,
        here / filename,
        here.parent / filename,
        here.parent.parent / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path.cwd() / filename


cache_file = nearby_cache_path(f"wikitext_bpe_cache_{dataset_range}_{vocab_size}.pkl")
tokenizer = BPETokenizer()

if cache_file.exists():
    print(f"Loading cached data from {cache_file}...")
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    data = cache_data['data']
    tokenizer.merges = cache_data['merges']
    tokenizer.vocab = cache_data['vocab']
else:
    if load_dataset is None:
        raise ImportError("datasets is required when the WikiText cache is not already available")

    print(f"Downloading and processing wikitext dataset (range: {dataset_range})...")
    textraw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    sample = textraw['train'].select(range(min(dataset_range, len(textraw['train']))))
    text = " ".join(sample["text"])

    print("Training regex BPE tokenizer...")
    tokenizer.train(text, vocab_size, verbose=True)

    print("Encoding dataset...")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    print(f"Saving cache to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump({'data': data, 'merges': tokenizer.merges, 'vocab': tokenizer.vocab}, f)


def decode(ids):
    return tokenizer.decode(ids)


def encode(text):
    return tokenizer.encode(text)


print("vocab size ", vocab_size)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)


def get_batch(split):
    data_src = train_data if split == 'train' else test_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i + block_size] for i in ix])
    y = torch.stack([data_src[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class RoutedMomentHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.scale = head_size**-0.5
        self.moment_top_k = moment_top_k

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.route_key = nn.Linear(n_embed, head_size, bias=False)
        self.route_query = nn.Linear(n_embed, head_size, bias=False)
        self.moment_value = nn.Linear(n_embed, head_size, bias=False)
        self.moment_mix = nn.Linear(4 * head_size, head_size)
        self.moment_gate = nn.Linear(n_embed, head_size)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def gather_context(self, x, idx):
        b, t, d = x.shape
        k = idx.size(-1)
        x_expanded = x.unsqueeze(1).expand(b, t, t, d)
        idx_expanded = idx.unsqueeze(-1).expand(b, t, k, d)
        return torch.gather(x_expanded, 2, idx_expanded)

    def weighted_moments(self, values, weights):
        weights = weights.unsqueeze(-1)
        mean = (weights * values).sum(dim=2)
        centered = values - mean.unsqueeze(2)
        variance = (weights * centered.pow(2)).sum(dim=2)
        third = (weights * centered.pow(3)).sum(dim=2)
        fourth = (weights * centered.pow(4)).sum(dim=2)

        std = torch.sqrt(variance + 1e-5)
        skew = third / (std.pow(3) + 1e-5)
        kurtosis = fourth / (variance.pow(2) + 1e-5)
        return torch.cat([mean, variance, skew, kurtosis], dim=-1)

    def forward(self, x):
        b, t, _ = x.shape
        causal_mask = self.tril[:t, :t].bool()
        moment_mask = torch.tril(self.tril[:t, :t], diagonal=-1).bool()
        moment_mask[0, 0] = True

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * self.scale
        wei = wei.masked_fill(causal_mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        base_out = wei @ v

        route_q = self.route_query(x)
        route_k = self.route_key(x)
        route_scores = route_q @ route_k.transpose(-2, -1) * self.scale
        route_scores = route_scores.masked_fill(moment_mask == 0, float('-inf'))

        routed_k = min(self.moment_top_k, t)
        top_scores, top_idx = torch.topk(route_scores, routed_k, dim=-1)
        moment_weights = F.softmax(top_scores, dim=-1)
        moment_weights = self.dropout(moment_weights)

        moment_values = self.gather_context(self.moment_value(x), top_idx)
        moment_summary = self.weighted_moments(moment_values, moment_weights)
        moment_out = self.moment_mix(moment_summary)

        gate = torch.sigmoid(self.moment_gate(x))
        return base_out + gate * moment_out


class MultiheadMomentAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([RoutedMomentHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MomentAttentionBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadMomentAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[MomentAttentionBlock(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = pos_embed + token_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    model = Transformer()
    total_params = sum(p.numel() for p in model.parameters())
    print('size of model', total_params)
    m = model.to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    start_time = time.time()

    for iter in range(max_iters):
        if not iter % eval_interval:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    generate_tokens = int(os.getenv('GENERATE_TOKENS', '200'))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=generate_tokens)[0].tolist()))

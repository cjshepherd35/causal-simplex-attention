# Difference from karpathy/bpekarpathy.py:
# 1. Replaces vanilla self-attention with routed 3-simplex attention.
# 2. A causal router picks the top-k available context tokens for each query.
# 3. The high-dimensional branch builds 3-token simplex states from routed triples,
#    then blends that branch back into ordinary causal attention with a learned gate.

from pathlib import Path
import os

import torch
import torch.nn as nn
from torch.nn import functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def find_input_path(filename='input.txt'):
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
    raise FileNotFoundError(f"could not find {filename} in cwd, script dir, or nearby parents")


with open(find_input_path(), 'r', encoding='utf-8') as f:
    text = f.read()
print('device is: ', device)

# parameters to tweak
max_iters = int(os.getenv('MAX_ITERS', '5001'))
eval_iters = int(os.getenv('EVAL_ITERS', '100'))
eval_interval = int(os.getenv('EVAL_INTERVAL', '1000'))
n_embed = int(os.getenv('N_EMBED', '64'))
block_size = int(os.getenv('BLOCK_SIZE', '64'))
batch_size = int(os.getenv('BATCH_SIZE', '12'))
learning_rate = float(os.getenv('LEARNING_RATE', '5e-3'))
n_head = int(os.getenv('N_HEAD', '4'))
n_layer = int(os.getenv('N_LAYER', '6'))
dropout = float(os.getenv('DROPOUT', '0.2'))
simplex_top_k = int(os.getenv('SIMPLEX_TOP_K', '5'))

vocab_size = int(os.getenv('VOCAB_SIZE', '400'))
num_merges = vocab_size - 256

assert vocab_size >= 256, "vocab_size must include the 256 raw byte tokens"
assert n_embed % n_head == 0, "n_embed must divide evenly across n_head"
assert simplex_top_k > 0, "simplex_top_k must be positive"

tokens = text.encode("utf-8")
tokens = list(map(int, tokens))


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
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


ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    if not stats:
        break
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("merged")
print('len: ', len(ids))

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
for idx in range(256, vocab_size):
    vocab.setdefault(idx, b'?')


def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors='replace')
    return text


def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)


def get_batch(split):
    # generate a small batch of data of inputs x and y
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


class RoutedThreeSimplexHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.scale = head_size**-0.5
        self.simplex_top_k = simplex_top_k

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.route_key = nn.Linear(n_embed, head_size, bias=False)
        self.route_query = nn.Linear(n_embed, head_size, bias=False)

        self.simplex_key = nn.Linear(n_embed, head_size, bias=False)
        self.simplex_value = nn.Linear(n_embed, head_size, bias=False)
        self.simplex_query_a = nn.Linear(n_embed, head_size, bias=False)
        self.simplex_query_b = nn.Linear(n_embed, head_size, bias=False)
        self.simplex_query_c = nn.Linear(n_embed, head_size, bias=False)
        self.simplex_value_a = nn.Linear(head_size, head_size, bias=False)
        self.simplex_value_b = nn.Linear(head_size, head_size, bias=False)
        self.simplex_value_c = nn.Linear(head_size, head_size, bias=False)
        self.simplex_interaction = nn.Linear(head_size, head_size, bias=False)
        self.simplex_out = nn.Linear(head_size, head_size, bias=False)
        self.simplex_gate = nn.Linear(n_embed, head_size)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def gather_context(self, x, idx):
        b, t, d = x.shape
        k = idx.size(-1)
        x_expanded = x.unsqueeze(1).expand(b, t, t, d)
        idx_expanded = idx.unsqueeze(-1).expand(b, t, k, d)
        return torch.gather(x_expanded, 2, idx_expanded)

    def forward(self, x):
        b, t, c = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        causal_mask = self.tril[:t, :t].bool()
        simplex_mask = torch.tril(self.tril[:t, :t], diagonal=-1).bool()
        simplex_mask[0, 0] = True

        wei = q @ k.transpose(-2, -1) * self.scale
        wei = wei.masked_fill(causal_mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        base_out = wei @ v

        route_q = self.route_query(x)
        route_k = self.route_key(x)
        route_scores = route_q @ route_k.transpose(-2, -1) * self.scale
        # The simplex branch routes over strict past tokens; token 0 gets a self fallback.
        route_scores = route_scores.masked_fill(simplex_mask == 0, float('-inf'))

        routed_k = min(self.simplex_top_k, t)
        top_scores, top_idx = torch.topk(route_scores, routed_k, dim=-1)

        simplex_k = self.gather_context(self.simplex_key(x), top_idx)
        simplex_v = self.gather_context(self.simplex_value(x), top_idx)

        qa = self.simplex_query_a(x)
        qb = self.simplex_query_b(x)
        qc = self.simplex_query_c(x)

        score_a = (qa.unsqueeze(2) * simplex_k).sum(dim=-1) * self.scale + top_scores
        score_b = (qb.unsqueeze(2) * simplex_k).sum(dim=-1) * self.scale + top_scores
        score_c = (qc.unsqueeze(2) * simplex_k).sum(dim=-1) * self.scale + top_scores

        simplex_logits = (
            score_a.unsqueeze(3).unsqueeze(4)
            + score_b.unsqueeze(2).unsqueeze(4)
            + score_c.unsqueeze(2).unsqueeze(3)
        )
        simplex_weights = F.softmax(simplex_logits.reshape(b, t, -1), dim=-1)
        simplex_weights = self.dropout(simplex_weights).view(b, t, routed_k, routed_k, routed_k)

        va = self.simplex_value_a(simplex_v)
        vb = self.simplex_value_b(simplex_v)
        vc = self.simplex_value_c(simplex_v)

        vertex_a = va.unsqueeze(3).unsqueeze(4)
        vertex_b = vb.unsqueeze(2).unsqueeze(4)
        vertex_c = vc.unsqueeze(2).unsqueeze(3)

        simplex_value = (vertex_a + vertex_b + vertex_c) / 3.0
        simplex_value = simplex_value + self.simplex_interaction(
            torch.tanh(vertex_a * vertex_b * vertex_c)
        )
        simplex_out = (simplex_weights.unsqueeze(-1) * simplex_value).sum(dim=(2, 3, 4))
        simplex_out = self.simplex_out(simplex_out)

        gate = torch.sigmoid(self.simplex_gate(x))
        out = base_out + gate * simplex_out
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([RoutedThreeSimplexHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


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


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size)
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
        # each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
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

    for iter in range(max_iters):
        # every once in awhile evaluate the loss on train and val sets
        if not iter % eval_interval:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    generate_tokens = int(os.getenv('GENERATE_TOKENS', '200'))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=generate_tokens)[0].tolist()))

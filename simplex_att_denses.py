import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('device is: ', device)

# parameters to tweak
max_iters = 8001
eval_iters = 100
eval_interval = 1000
n_embed = 128
block_size = 32
batch_size = 16
learning_rate = 1e-3
n_head = 4
n_layer = 6
dropout = 0.2

vocab_size = 400
num_merges = vocab_size - 256

# --- Tokenization (BPE) ---
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
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
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
    if not stats: break
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors='replace')

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

def get_batch(split):
    data_src = train_data if split == 'train' else test_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
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

# --- Model Components ---

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SimplexHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.wk_prime = nn.Linear(head_size, head_size, bias=False)
        self.wv_prime = nn.Linear(head_size, head_size, bias=False)
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scale = self.head_size**-0.5

        # Vectorized stable Simplex Attention
        s = (q @ k.transpose(-2, -1)) * scale 
        prefix_mask = torch.tril(torch.ones(t, t, device=x.device), diagonal=-1)
        safe_mask = prefix_mask.clone()
        safe_mask[0, 0] = 1
        
        s_3d = s.unsqueeze(1).masked_fill(safe_mask.unsqueeze(1) == 0, float('-inf'))
        a_past = F.softmax(s_3d, dim=-1) 
        h_all = a_past @ v.unsqueeze(1) 
        
        k_graph, v_graph = self.wk_prime(h_all), self.wv_prime(h_all)
        s_curr = torch.einsum('btd, btjd -> btj', q, k_graph) * scale
        s_curr = s_curr.masked_fill(safe_mask == 0, float('-inf'))
        
        wei_curr = F.softmax(s_curr, dim=-1)
        wei_curr = self.dropout(wei_curr)
        
        out = torch.einsum('btj, btjd -> btd', wei_curr, v_graph)
        out[:, 0, :] = 0 # First token sees nothing
        return out

class SimplexMultiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SimplexHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FullSimplexBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = SimplexMultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DenseBlock(nn.Module):
    """A purely dense block (no attention, token-wise only)."""
    def __init__(self, n_embed):
        super().__init__()
        self.ffwd = FeedForward(n_embed)
        self.ln = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.ffwd(self.ln(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # 1st Layer: Simplex Communication
        # Layers 2-6: Purely Dense
        layers = [FullSimplexBlock(n_embed, n_head)]
        for _ in range(n_layer - 1):
            layers.append(DenseBlock(n_embed))
        self.blocks = nn.Sequential(*layers)
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) 

    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(t, device=device))
        x = tok_emb + pos_emb
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
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = Transformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))

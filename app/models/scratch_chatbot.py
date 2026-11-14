"""Transformer decoder estilo GPT implementado desde cero (sin librerías de modelos de lenguaje)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Atención multi-cabeza con máscara causal: cada token solo ve los anteriores."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)

        # Máscara causal (triángulo inferior). Forma parte del checkpoint, por eso se mantiene en 1024.
        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class DecoderBlock(nn.Module):
    """Bloque GPT: atención causal + MLP, ambos con conexión residual y LayerNorm previo."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MedicalGPTFromScratch(nn.Module):
    """Embeddings de token y posición + N bloques decoder + cabeza de lenguaje."""
    def __init__(self, vocab_size, embed_dim=128, depth=3, num_heads=4, max_seq_len=64):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.transformer_drop = nn.Dropout(0.1)
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList([DecoderBlock(embed_dim, num_heads) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        _, t = idx.size()
        assert t <= self.max_seq_len, f"Secuencia de {t} supera el límite de {self.max_seq_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)

        x = self.transformer_drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.3):
        """Generación autorregresiva: muestrea un token cada vez y lo añade a la secuencia."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

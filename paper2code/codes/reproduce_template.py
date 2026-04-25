#!/usr/bin/env python3
"""
Self-contained Transformer reproduction script.
Paper: Attention Is All You Need (Vaswani et al., 2017)
This demo uses scaled-down hyperparams and a data subset for fast iteration.
"""

import argparse
import json
import math
import os
import sys
import time

# Flush stdout immediately so progress shows in correct order
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--reproduce_dir", type=str, default=".")
parser.add_argument("--max_train_steps", type=int, default=500)
parser.add_argument("--dataset_size", type=int, default=5000)
# These are filled in by 5_reproduce.py via string substitution
parser.add_argument("--hf_dataset_id", type=str, default="wmt14")
parser.add_argument("--hf_dataset_cfg", type=str, default="de-en")
parser.add_argument("--src_lang", type=str, default="en")
parser.add_argument("--tgt_lang", type=str, default="de")
parser.add_argument("--vocab_size", type=int, default=8000)
parser.add_argument("--paper_target_bleu", type=float, default=27.3)
args = parser.parse_args()

REPRODUCE_DIR = args.reproduce_dir
MAX_TRAIN_STEPS = args.max_train_steps
DATASET_SIZE = args.dataset_size
HF_DATASET_ID = args.hf_dataset_id
HF_DATASET_CFG = args.hf_dataset_cfg
SRC_LANG = args.src_lang
TGT_LANG = args.tgt_lang
VOCAB_SIZE = min(args.vocab_size, 8000)  # cap for demo speed
PAPER_TARGET_BLEU = args.paper_target_bleu

# Demo-scale model (paper: d_model=512, 8 heads, 6 layers)
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 3
D_FF = 512
DROPOUT = 0.1
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
WARMUP_STEPS = 400
LABEL_SMOOTHING = 0.1
BEAM_SIZE = 4
LENGTH_PENALTY_ALPHA = 0.6

os.makedirs(REPRODUCE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_raw_data():
    from datasets import load_dataset
    print(f"Downloading {HF_DATASET_ID} ({HF_DATASET_CFG})...")
    ds = load_dataset(HF_DATASET_ID, HF_DATASET_CFG, trust_remote_code=True)

    def extract(split, n=None):
        rows = ds[split]["translation"]
        if n:
            rows = rows[:n]
        return [r[SRC_LANG] for r in rows], [r[TGT_LANG] for r in rows]

    train_src, train_tgt = extract("train", DATASET_SIZE)
    val_src, val_tgt = extract("validation")
    test_src, test_tgt = extract("test")
    print(f"Train: {len(train_src)} | Val: {len(val_src)} | Test: {len(test_src)}")
    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt


def train_tokenizer(train_src, train_tgt):
    import sentencepiece as spm
    tok_prefix = os.path.join(REPRODUCE_DIR, "tokenizer")
    model_path = tok_prefix + ".model"
    if os.path.exists(model_path):
        print("Tokenizer exists, loading...")
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        return sp
    corpus_path = os.path.join(REPRODUCE_DIR, "corpus.txt")
    with open(corpus_path, "w") as f:
        for s in train_src + train_tgt:
            f.write(s.strip() + "\n")
    print(f"Training BPE tokenizer (vocab={VOCAB_SIZE})...")
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=tok_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        character_coverage=0.9995,
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


class TranslationDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, sp, max_len=MAX_SEQ_LEN):
        pairs = []
        for s, t in zip(src_sents, tgt_sents):
            s_ids = [2] + sp.EncodeAsIds(s)[:max_len - 2] + [3]
            t_ids = [2] + sp.EncodeAsIds(t)[:max_len - 2] + [3]
            pairs.append((s_ids, t_ids))
        pairs.sort(key=lambda x: len(x[0]) + len(x[1]))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def collate(batch, pad=0):
    src_batch, tgt_batch = zip(*batch)
    max_s = max(len(s) for s in src_batch)
    max_t = max(len(t) for t in tgt_batch)
    src_t = torch.tensor(
        [s + [pad] * (max_s - len(s)) for s in src_batch], dtype=torch.long
    )
    tgt_t = torch.tensor(
        [t + [pad] * (max_t - len(t)) for t in tgt_batch], dtype=torch.long
    )
    return src_t, tgt_t


# ── Model ─────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        def proj(lin, x):
            return lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)

        Q, K, V = proj(self.wq, q), proj(self.wk, k), proj(self.wv, v)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))
        x = (
            torch.matmul(attn, V)
            .transpose(1, 2)
            .contiguous()
            .view(B, -1, self.h * self.d_k)
        )
        return self.wo(x)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.drop(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc, src_mask, tgt_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc, enc, src_mask)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = SinusoidalPE(d_model, max_len)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm_enc = nn.LayerNorm(d_model)
        self.norm_dec = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.proj.weight = self.emb.weight  # weight tying (Section 3.4)
        self.d_model = d_model
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.drop(self.pe(self.emb(src) * math.sqrt(self.d_model)))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return self.norm_enc(x)

    def decode(self, tgt, enc, src_mask, tgt_mask):
        x = self.drop(self.pe(self.emb(tgt) * math.sqrt(self.d_model)))
        for layer in self.dec_layers:
            x = layer(x, enc, src_mask, tgt_mask)
        return self.norm_dec(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc = self.encode(src, src_mask)
        dec = self.decode(tgt, enc, src_mask, tgt_mask)
        return self.proj(dec)


def make_src_mask(src, pad=0):
    return (src == pad).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad=0):
    T = tgt.size(1)
    pad_mask = (tgt == pad).unsqueeze(1).unsqueeze(2)
    causal = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
    return pad_mask | causal.unsqueeze(0)


# ── Training helpers ──────────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, eps=0.1, pad=0):
        super().__init__()
        self.eps = eps
        self.pad = pad
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        B, T, V = logits.shape
        logits = logits.reshape(-1, V)
        target = target.reshape(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        smooth = self.eps / (V - 2)
        one_hot = torch.full_like(log_probs, smooth)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0 - self.eps + smooth)
        one_hot[:, self.pad] = 0
        mask = target != self.pad
        loss = -(one_hot * log_probs).sum(dim=-1)
        return loss[mask].mean()


def lr_schedule(step, d_model, warmup):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


# ── Beam search ───────────────────────────────────────────────────────────────

def beam_search(model, src, sp, beam_size=4, alpha=0.6, max_len=None):
    model.train(False)
    if max_len is None:
        max_len = src.size(1) + 50
    src_mask = make_src_mask(src)
    with torch.no_grad():
        enc = model.encode(src, src_mask)
    BOS, EOS = 2, 3
    beams = [([BOS], 0.0)]
    completed = []
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq[-1] == EOS:
                completed.append((seq, score))
                continue
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_mask = make_tgt_mask(tgt)
            with torch.no_grad():
                logits = model.proj(model.decode(tgt, enc, src_mask, tgt_mask))
            log_probs = F.log_softmax(logits[0, -1], dim=-1)
            topk = log_probs.topk(beam_size)
            for log_p, tok in zip(topk.values, topk.indices):
                candidates.append((seq + [tok.item()], score + log_p.item()))
        candidates.sort(
            key=lambda x: x[1] / ((len(x[0]) ** alpha) / (6 ** alpha)), reverse=True
        )
        beams = candidates[:beam_size]
        if all(s[-1] == EOS for s, _ in beams):
            completed.extend(beams)
            break
    if not completed:
        completed = beams
    completed.sort(
        key=lambda x: x[1] / ((len(x[0]) ** alpha) / (6 ** alpha)), reverse=True
    )
    best = completed[0][0]
    return sp.DecodeIds([t for t in best if t not in (BOS, EOS)])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Transformer Reproduction ===")
    print(f"Paper target BLEU : {PAPER_TARGET_BLEU}  (newstest2014, base EN-DE)")
    print(f"Demo              : {MAX_TRAIN_STEPS} steps, {DATASET_SIZE} training pairs\n")

    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = load_raw_data()
    sp = train_tokenizer(train_src, train_tgt)
    actual_vocab = sp.GetPieceSize()
    print(f"Vocab size: {actual_vocab}")

    train_ds = TranslationDataset(train_src, train_tgt, sp)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True
    )

    model = Transformer(
        actual_vocab, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT, MAX_SEQ_LEN
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}  (paper base: ~65M)\n")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_schedule(step, D_MODEL, WARMUP_STEPS)
    )
    criterion = LabelSmoothingLoss(actual_vocab, eps=LABEL_SMOOTHING)

    print("Training...")
    step = 0
    ckpt_path = os.path.join(REPRODUCE_DIR, "checkpoint.pt")
    train_iter = iter(train_loader)
    t0 = time.time()

    while step < MAX_TRAIN_STEPS:
        model.train()
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt = next(train_iter)

        src, tgt = src.to(device), tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits, tgt_out)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 100 == 0:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step:5d} | loss {loss.item():.4f} | lr {lr_now:.2e} | {elapsed:.0f}s")
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nTraining complete ({step} steps). Running evaluation on test set...")

    n_eval = min(200, len(test_src))
    hypotheses, references = [], []
    for i in range(n_eval):
        src_ids = [2] + sp.EncodeAsIds(test_src[i])[: MAX_SEQ_LEN - 2] + [3]
        src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
        hyp = beam_search(model, src_t, sp, beam_size=BEAM_SIZE, alpha=LENGTH_PENALTY_ALPHA)
        hypotheses.append(hyp)
        references.append(test_tgt[i])
        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i+1}/{n_eval}")

    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    bleu_score = round(bleu.score, 2)

    print("\n" + "=" * 50)
    print(f"  BLEU ({n_eval} test sentences): {bleu_score}")
    print(f"  Paper target BLEU:            {PAPER_TARGET_BLEU}")
    print(f"  Steps used  : {MAX_TRAIN_STEPS} / 100,000 (paper)")
    print(f"  Data used   : {DATASET_SIZE:,} / 4,500,000 (paper)")
    print(f"  Model size  : {n_params:,} / ~65,000,000 (paper)")
    scale = 100_000 // max(MAX_TRAIN_STEPS, 1)
    print(f"  (Demo used {scale}x fewer steps — lower BLEU is expected)")
    print("=" * 50)

    results = {
        "bleu_score": bleu_score,
        "paper_target_bleu": PAPER_TARGET_BLEU,
        "training_steps": MAX_TRAIN_STEPS,
        "dataset_size": DATASET_SIZE,
        "model_params": n_params,
        "num_eval_sentences": n_eval,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
    }
    results_path = os.path.join(REPRODUCE_DIR, "reproduce_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

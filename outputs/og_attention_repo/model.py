## model.py
"""Model components for reproducing 'Neural Machine Translation by Jointly Learning to Align and Translate'.

This module implements the exact architectures described in the paper:
- GatedRNNCell: Cho et al. (2014a) gated hidden unit (pre-GRU).
- AlignmentModel: feedforward alignment model for attention.
- MaxoutOutputLayer: output layer with a single maxout hidden layer and softmax.
- RNNEncDec: baseline RNN Encoder‑Decoder (Cho et al., 2014a).
- RNNsearch: proposed model with bidirectional encoder and attention‑based decoder.

All hyperparameters are set to the values specified in the paper and the accompanying config.yaml.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedRNNCell(nn.Module):
    """Gated RNN cell from Cho et al. (2014a) (a precursor to GRU).

    Used for all RNN components (encoder/decoder) in both models.
    Equations are given in Appendix A.1.1 of the paper.

    Attributes:
        input_dim (int): Dimension of input embeddings.
        hidden_dim (int): Hidden state size (1000 in paper).
        context_dim (Optional[int]): Dimension of context vector, if used (for decoder cells).
    """

    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # Input weights (W, W_z, W_r)
        self.W = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_z = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_r = nn.Parameter(torch.Tensor(hidden_dim, input_dim))

        # Recurrent weights (U, U_z, U_r) – initialized orthogonal
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.U_z = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.U_r = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))

        # Context weights (C, C_z, C_r) – only if context is provided
        if context_dim is not None:
            self.C = nn.Parameter(torch.Tensor(hidden_dim, context_dim))
            self.C_z = nn.Parameter(torch.Tensor(hidden_dim, context_dim))
            self.C_r = nn.Parameter(torch.Tensor(hidden_dim, context_dim))
        else:
            self.register_parameter('C', None)
            self.register_parameter('C_z', None)
            self.register_parameter('C_r', None)

        # Biases (all zero)
        self.b = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_z = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_r = nn.Parameter(torch.Tensor(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights as specified in Appendix B.1 of the paper."""
        # Recurrent weights: random orthogonal matrices
        for weight in (self.U, self.U_z, self.U_r):
            nn.init.orthogonal_(weight)

        # Other weights (input, context): Gaussian with mean 0, std = 0.01 (since variance = 0.01²)
        std = 0.01
        for weight in (self.W, self.W_z, self.W_r):
            nn.init.normal_(weight, mean=0.0, std=std)
        if self.context_dim is not None:
            nn.init.normal_(self.C, mean=0.0, std=std)
            nn.init.normal_(self.C_z, mean=0.0, std=std)
            nn.init.normal_(self.C_r, mean=0.0, std=std)

        # Biases zero
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.b_z)
        nn.init.zeros_(self.b_r)

    def forward(self, x: torch.Tensor, s_prev: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """Compute next hidden state.

        Args:
            x: Input embedding, shape (batch_size, input_dim).
            s_prev: Previous hidden state, shape (batch_size, hidden_dim).
            c: Optional context vector, shape (batch_size, context_dim).

        Returns:
            New hidden state, shape (batch_size, hidden_dim).
        """
        # Reset gate
        r = torch.sigmoid(
            F.linear(x, self.W_r) + F.linear(s_prev, self.U_r) +
            (F.linear(c, self.C_r) if c is not None and self.C_r is not None else 0) +
            self.b_r
        )
        # Update gate
        z = torch.sigmoid(
            F.linear(x, self.W_z) + F.linear(s_prev, self.U_z) +
            (F.linear(c, self.C_z) if c is not None and self.C_z is not None else 0) +
            self.b_z
        )
        # Candidate state
        s_prev_modulated = r * s_prev
        candidate = torch.tanh(
            F.linear(x, self.W) + F.linear(s_prev_modulated, self.U) +
            (F.linear(c, self.C) if c is not None and self.C is not None else 0) +
            self.b
        )
        # New state
        s_new = (1 - z) * s_prev + z * candidate
        return s_new


class AlignmentModel(nn.Module):
    """Feedforward alignment model for computing attention scores.

    Computes a score e_ij = v_a^T tanh(W_a * s_{i-1} + U_a * h_j).
    See Appendix A.1.2.

    Attributes:
        hidden_dim (int): Decoder hidden size (1000).
        context_dim (int): Annotation dimension (2000 for RNNsearch).
    """

    def __init__(self, hidden_dim: int = 1000, context_dim: int = 2000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # W_a: (hidden_dim, hidden_dim)
        self.W_a = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        # U_a: (hidden_dim, context_dim)
        self.U_a = nn.Parameter(torch.Tensor(hidden_dim, context_dim))
        # v_a: (hidden_dim,) – vector for dot product
        self.v_a = nn.Parameter(torch.Tensor(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize as per Appendix B.1: W_a, U_a ~ N(0, 0.001²), v_a = 0."""
        std = 0.001
        nn.init.normal_(self.W_a, mean=0.0, std=std)
        nn.init.normal_(self.U_a, mean=0.0, std=std)
        nn.init.zeros_(self.v_a)

    def forward(self, s_prev: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """Compute alignment score(s) for given decoder state and annotation(s).

        Args:
            s_prev: Decoder previous hidden state, shape (batch_size, hidden_dim) or (hidden_dim,).
            h_j: Annotation(s), shape (batch_size, context_dim) for a single annotation,
                 or (batch_size, num_annotations, context_dim) for multiple.

        Returns:
            Scores: scalar or tensor of shape (batch_size,) or (batch_size, num_annotations).
        """
        # Ensure s_prev is at least 2D
        if s_prev.dim() == 1:
            s_prev = s_prev.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        # W_a * s_prev -> (batch_size, hidden_dim)
        W_a_s = F.linear(s_prev, self.W_a)

        if h_j.dim() == 3:  # multiple annotations
            # h_j: (batch, num_annotations, context_dim)
            # U_a * h_j -> (batch, num_annotations, hidden_dim)
            U_a_h = torch.matmul(h_j, self.U_a.t())  # (batch, num, hidden_dim)
            # Expand W_a_s to match
            W_a_s_exp = W_a_s.unsqueeze(1).expand(-1, h_j.size(1), -1)  # (batch, num, hidden_dim)
            energy = torch.tanh(W_a_s_exp + U_a_h)  # (batch, num, hidden_dim)
            # Dot with v_a -> (batch, num)
            scores = torch.matmul(energy, self.v_a)  # (batch, num)
        else:  # single annotation
            if h_j.dim() == 1:
                h_j = h_j.unsqueeze(0)
            U_a_h = F.linear(h_j, self.U_a)  # (batch, hidden_dim)
            energy = torch.tanh(W_a_s + U_a_h)  # (batch, hidden_dim)
            scores = torch.matmul(energy, self.v_a)  # (batch,)

        if unsqueezed:
            scores = scores.squeeze(0) if scores.dim() > 1 else scores
        return scores


class MaxoutOutputLayer(nn.Module):
    """Output layer with a single maxout hidden layer and softmax.

    Computes logits for the target vocabulary as described in Appendix A.1.1.
    t_i = U_o * s_{i-1} + V_o * y_{i-1}_emb + C_o * c_i,
    maxout over (t_i split into pairs), then linear projection to vocab size.

    Attributes:
        hidden_dim (int): Decoder hidden size (1000).
        maxout_dim (int): Maxout output dimension (500, so 2*maxout_dim = 1000 input to maxout).
        vocab_size (int): Target vocabulary size (30000).
        emb_dim (int): Embedding dimension of previous target word (620).
        context_dim (int): Dimension of context vector (1000 for RNNEncDec, 2000 for RNNsearch).
    """

    def __init__(self, hidden_dim: int = 1000, maxout_dim: int = 500, vocab_size: int = 30000,
                 emb_dim: int = None, context_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.maxout_dim = maxout_dim
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_dim = context_dim

        if emb_dim is None or context_dim is None:
            raise ValueError("emb_dim and context_dim must be provided to MaxoutOutputLayer.")

        # Weights
        self.U_o = nn.Parameter(torch.Tensor(2 * maxout_dim, hidden_dim))
        self.V_o = nn.Parameter(torch.Tensor(2 * maxout_dim, emb_dim))
        self.C_o = nn.Parameter(torch.Tensor(2 * maxout_dim, context_dim))
        self.W_o = nn.Parameter(torch.Tensor(vocab_size, maxout_dim))  # (vocab_size, maxout_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights as per Appendix B.1: Gaussian with mean 0, std = 0.01."""
        std = 0.01
        nn.init.normal_(self.U_o, mean=0.0, std=std)
        nn.init.normal_(self.V_o, mean=0.0, std=std)
        nn.init.normal_(self.C_o, mean=0.0, std=std)
        nn.init.normal_(self.W_o, mean=0.0, std=std)

    def forward(self, s_prev: torch.Tensor, y_prev_emb: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute logits over target vocabulary.

        Args:
            s_prev: Previous decoder hidden state, shape (batch_size, hidden_dim).
            y_prev_emb: Embedding of previous target word, shape (batch_size, emb_dim).
            c: Context vector, shape (batch_size, context_dim).

        Returns:
            Logits, shape (batch_size, vocab_size).
        """
        # Combined feature
        t = (F.linear(s_prev, self.U_o) +
             F.linear(y_prev_emb, self.V_o) +
             F.linear(c, self.C_o))  # (batch, 2*maxout_dim)
        # Maxout: reshape to (batch, maxout_dim, 2) and take max
        t = t.view(t.size(0), self.maxout_dim, 2)
        t_max = torch.max(t, dim=2)[0]  # (batch, maxout_dim)
        # Project to vocabulary
        logits = F.linear(t_max, self.W_o)  # (batch, vocab_size)
        return logits


class RNNEncDec(nn.Module):
    """Baseline RNN Encoder‑Decoder model (Cho et al., 2014a).

    Uses a single forward RNN encoder and a decoder with a fixed context vector.
    """

    def __init__(self, src_vocab_size: int = 30000, tgt_vocab_size: int = 30000,
                 emb_dim: int = 620, hidden_dim: int = 1000, maxout_dim: int = 500):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.maxout_dim = maxout_dim

        # Embeddings
        self.src_emb = nn.Embedding(src_vocab_size, emb_dim)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim)

        # Encoder: forward RNN only, no context
        self.encoder_cell = GatedRNNCell(input_dim=emb_dim, hidden_dim=hidden_dim, context_dim=None)
        # Decoder: uses fixed context vector of size hidden_dim
        self.decoder_cell = GatedRNNCell(input_dim=emb_dim, hidden_dim=hidden_dim, context_dim=hidden_dim)

        # Output layer: context_dim = hidden_dim (fixed)
        self.output_layer = MaxoutOutputLayer(
            hidden_dim=hidden_dim, maxout_dim=maxout_dim, vocab_size=tgt_vocab_size,
            emb_dim=emb_dim, context_dim=hidden_dim
        )

        # Initial decoder state weight: s0 = tanh(W_s * c)
        self.W_s = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize embeddings and W_s as per Appendix B.1."""
        std = 0.01
        nn.init.normal_(self.src_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.W_s, mean=0.0, std=std)

    def encode(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """Encode source sentence into a fixed context vector (last encoder hidden state).

        Args:
            src: Source token indices, shape (batch_size, src_len).
            src_lengths: Not used in this simple loop‑based encoder.

        Returns:
            Context vector c, shape (batch_size, hidden_dim).
        """
        batch_size, src_len = src.size()
        src_embeds = self.src_emb(src)  # (batch, src_len, emb_dim)

        h_prev = torch.zeros(batch_size, self.hidden_dim, device=src.device)
        for t in range(src_len):
            x_t = src_embeds[:, t, :]
            h_prev = self.encoder_cell(x_t, h_prev, c=None)
        c = h_prev  # last hidden state
        return c

    def decode(self, s_prev: torch.Tensor, y_prev: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode one step.

        Args:
            s_prev: Previous decoder hidden state, shape (batch_size, hidden_dim).
            y_prev: Previous target token indices, shape (batch_size,).
            c: Fixed context vector, shape (batch_size, hidden_dim).

        Returns:
            logits: (batch_size, vocab_size).
            s_new: New decoder hidden state, (batch_size, hidden_dim).
        """
        y_emb = self.tgt_emb(y_prev)  # (batch, emb_dim)
        s_new = self.decoder_cell(y_emb, s_prev, c)
        # Use s_prev (not s_new) for output as per paper
        logits = self.output_layer(s_prev, y_emb, c)
        return logits, s_new

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            src: Source token indices, shape (batch_size, src_len).
            tgt: Target token indices, shape (batch_size, tgt_len) – starts with <start>.
            src_lengths: Source lengths (unused).

        Returns:
            Logits for each target position, shape (batch_size, tgt_len, vocab_size).
        """
        batch_size, tgt_len = tgt.size()
        c = self.encode(src, src_lengths)  # (batch, hidden_dim)
        s0 = torch.tanh(F.linear(c, self.W_s))  # (batch, hidden_dim)

        logits_list = []
        s_prev = s0
        for i in range(tgt_len):
            y_prev = tgt[:, i]  # (batch,)
            logits_i, s_new = self.decode(s_prev, y_prev, c)
            logits_list.append(logits_i)
            s_prev = s_new

        logits = torch.stack(logits_list, dim=1)  # (batch, tgt_len, vocab_size)
        return logits


class RNNsearch(nn.Module):
    """Proposed model with bidirectional encoder and attention (RNNsearch).

    Encoder: bidirectional RNN producing annotations h_j = [forward_j; backward_j].
    Decoder: attention‑based, computes a context vector per step via alignment model.
    """

    def __init__(self, src_vocab_size: int = 30000, tgt_vocab_size: int = 30000,
                 emb_dim: int = 620, hidden_dim: int = 1000, maxout_dim: int = 500):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.maxout_dim = maxout_dim

        # Embeddings
        self.src_emb = nn.Embedding(src_vocab_size, emb_dim)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim)

        # Bidirectional encoder cells
        self.fwd_encoder_cell = GatedRNNCell(input_dim=emb_dim, hidden_dim=hidden_dim, context_dim=None)
        self.bwd_encoder_cell = GatedRNNCell(input_dim=emb_dim, hidden_dim=hidden_dim, context_dim=None)

        # Decoder cell: context is annotation of size 2*hidden_dim
        self.decoder_cell = GatedRNNCell(input_dim=emb_dim, hidden_dim=hidden_dim, context_dim=2*hidden_dim)

        # Alignment model
        self.alignment_model = AlignmentModel(hidden_dim=hidden_dim, context_dim=2*hidden_dim)

        # Output layer: context_dim = 2*hidden_dim
        self.output_layer = MaxoutOutputLayer(
            hidden_dim=hidden_dim, maxout_dim=maxout_dim, vocab_size=tgt_vocab_size,
            emb_dim=emb_dim, context_dim=2*hidden_dim
        )

        # Initial decoder state weight: s0 = tanh(W_s * backward_first)
        self.W_s = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize embeddings and W_s."""
        std = 0.01
        nn.init.normal_(self.src_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.W_s, mean=0.0, std=std)

    def encode(self, src: torch.Tensor, src_lengths: torch.Tensor) -> List[torch.Tensor]:
        """Encode source sentence into a list of annotations.

        Args:
            src: Source token indices, shape (batch_size, src_len).
            src_lengths: Unused.

        Returns:
            List of annotations, each of shape (batch_size, 2*hidden_dim), length src_len.
        """
        batch_size, src_len = src.size()
        src_embeds = self.src_emb(src)  # (batch, src_len, emb_dim)

        # Forward RNN
        h_fwd_prev = torch.zeros(batch_size, self.hidden_dim, device=src.device)
        forward_states = []
        for t in range(src_len):
            x_t = src_embeds[:, t, :]
            h_fwd_prev = self.fwd_encoder_cell(x_t, h_fwd_prev, c=None)
            forward_states.append(h_fwd_prev)

        # Backward RNN (from end to start)
        h_bwd_prev = torch.zeros(batch_size, self.hidden_dim, device=src.device)
        backward_states = [None] * src_len
        for idx in range(src_len - 1, -1, -1):
            x = src_embeds[:, idx, :]
            h_bwd_prev = self.bwd_encoder_cell(x, h_bwd_prev, c=None)
            backward_states[idx] = h_bwd_prev

        # Concatenate forward and backward states for each position
        annotations = []
        for j in range(src_len):
            h_j = torch.cat([forward_states[j], backward_states[j]], dim=-1)  # (batch, 2*hidden_dim)
            annotations.append(h_j)

        return annotations

    def decode(self, s_prev: torch.Tensor, y_prev: torch.Tensor,
               annotations: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one step with attention.

        Args:
            s_prev: Previous decoder hidden state, shape (batch_size, hidden_dim).
            y_prev: Previous target token indices, shape (batch_size,).
            annotations: List of annotation tensors, each (batch_size, 2*hidden_dim).

        Returns:
            logits: (batch_size, vocab_size).
            s_new: New decoder hidden state, (batch_size, hidden_dim).
            alpha: Alignment weights, shape (batch_size, src_len).
        """
        batch_size = s_prev.size(0)
        src_len = len(annotations)
        y_emb = self.tgt_emb(y_prev)  # (batch, emb_dim)

        # Stack annotations into a tensor for efficient computation
        annot_tensor = torch.stack(annotations, dim=1)  # (batch, src_len, 2*hidden_dim)

        # Precompute U_a * h_j for all annotations (independent of decoder state)
        # U_a: (hidden_dim, 2*hidden_dim) -> annot_tensor @ U_a.t() -> (batch, src_len, hidden_dim)
        U_a_h = torch.matmul(annot_tensor, self.alignment_model.U_a.t())  # (batch, src_len, hidden_dim)

        # Compute W_a * s_prev
        W_a_s = F.linear(s_prev, self.alignment_model.W_a)  # (batch, hidden_dim)
        W_a_s_exp = W_a_s.unsqueeze(1).expand(-1, src_len, -1)  # (batch, src_len, hidden_dim)

        # Energy and scores
        energy = torch.tanh(W_a_s_exp + U_a_h)  # (batch, src_len, hidden_dim)
        scores = torch.matmul(energy, self.alignment_model.v_a)  # (batch, src_len)

        # Alignment weights
        alpha = F.softmax(scores, dim=1)  # (batch, src_len)

        # Context vector: weighted sum of annotations
        alpha_unsqueezed = alpha.unsqueeze(2)  # (batch, src_len, 1)
        c_i = torch.sum(alpha_unsqueezed * annot_tensor, dim=1)  # (batch, 2*hidden_dim)

        # New decoder state
        s_new = self.decoder_cell(y_emb, s_prev, c_i)

        # Logits (using s_prev as per paper)
        logits = self.output_layer(s_prev, y_emb, c_i)

        return logits, s_new, alpha

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            src: Source token indices, shape (batch_size, src_len).
            tgt: Target token indices, shape (batch_size, tgt_len) – starts with <start>.
            src_lengths: Source lengths (unused).

        Returns:
            Logits for each target position, shape (batch_size, tgt_len, vocab_size).
        """
        batch_size, tgt_len = tgt.size()
        annotations = self.encode(src, src_lengths)  # List of (batch, 2*hidden_dim)

        # Initial decoder state: s0 = tanh(W_s * backward_first)
        first_annotation = annotations[0]  # (batch, 2*hidden_dim)
        backward_first = first_annotation[:, self.hidden_dim:]  # (batch, hidden_dim)
        s0 = torch.tanh(F.linear(backward_first, self.W_s))  # (batch, hidden_dim)

        logits_list = []
        s_prev = s0
        for i in range(tgt_len):
            y_prev = tgt[:, i]  # (batch,)
            logits_i, s_new, _ = self.decode(s_prev, y_prev, annotations)
            logits_list.append(logits_i)
            s_prev = s_new

        logits = torch.stack(logits_list, dim=1)  # (batch, tgt_len, vocab_size)
        return logits

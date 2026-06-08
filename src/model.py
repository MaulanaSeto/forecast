"""
Temporal Fusion Transformer (TFT) — built from scratch in PyTorch.

This module contains every building block required by the TFT architecture
for multi-horizon time-series forecasting, plus two ablation variants.

Classes
-------
GRN               - Gated Residual Network
VSN               - Variable Selection Network
TemporalSelfAttention - Multi-head temporal self-attention
StaticEncoder     - Produces four context vectors from static features
TFT               - Full Temporal Fusion Transformer
TFT_NoVSN         - Ablation: replaces VSN with simple variable averaging
TFT_NoAttention   - Ablation: skips the self-attention block
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor


# 1. Gated Residual Network (GRN)
class GRN(nn.Module):
    """Gated Residual Network.

    Applies a two-layer feed-forward transformation with an optional
    additive context signal, a Gated Linear Unit (GLU) for output gating,
    a skip (residual) connection, and Layer Normalization.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the primary input tensor.
    hidden_dim : int
        Width of the internal hidden layer.
    output_dim : int
        Dimensionality of the output tensor.
    dropout : float, optional
        Dropout probability applied after the hidden layer (default 0.1).
    context_dim : int or None, optional
        If given, a linear projection of the context vector is added to
        the first linear projection of *x* before the activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Optional context injection
        self.context_proj: nn.Linear | None = None
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        self.hidden_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # GLU gate — projects to 2 * output_dim then splits
        self.gate_linear = nn.Linear(hidden_dim, output_dim * 2)

        # Skip connection (project if dimensions differ)
        self.skip_proj: nn.Linear | None = None
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Primary input of shape ``(..., input_dim)``.
        context : Tensor or None
            Optional context of shape ``(..., context_dim)``.

        Returns
        -------
        Tensor
            Output of shape ``(..., output_dim)``.
        """
        # --- skip connection source ---
        residual = x if self.skip_proj is None else self.skip_proj(x)

        # --- first projection + optional context ---
        h = self.input_proj(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)

        # --- ELU → hidden linear → dropout ---
        h = F.elu(h)
        h = self.hidden_linear(h)
        h = self.dropout(h)

        # --- GLU ---
        gate_input = self.gate_linear(h)
        value, gate = gate_input.chunk(2, dim=-1)
        h = value * torch.sigmoid(gate)

        # --- residual + LayerNorm ---
        return self.layer_norm(h + residual)


# 2. Variable Selection Network (VSN)
class VSN(nn.Module):
    """Variable Selection Network.

    Learns to softmax-weight and combine per-variable representations via
    individual GRNs and a shared weighting GRN.

    Parameters
    ----------
    n_vars : int
        Number of input variables.
    d_model : int
        Model / embedding dimensionality.
    dropout : float, optional
        Dropout probability used inside GRNs (default 0.1).
    context_dim : int or None, optional
        Dimensionality of the static context vector that conditions the
        variable-selection weights.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint

        # Per-variable transformation GRNs
        self.var_grns = nn.ModuleList(
            [
                GRN(
                    input_dim=d_model,
                    hidden_dim=d_model,
                    output_dim=d_model,
                    dropout=dropout,
                )
                for _ in range(n_vars)
            ]
        )

        # Weight GRN — produces one scalar weight per variable
        self.weight_grn = GRN(
            input_dim=n_vars * d_model,
            hidden_dim=d_model,
            output_dim=n_vars,
            dropout=dropout,
            context_dim=context_dim,
        )

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        # pylint: disable=invalid-name
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, time, n_vars, d_model)``.
        context : Tensor or None
            Static context of shape ``(batch, context_dim)``.

        Returns
        -------
        Tensor
            Weighted combination of shape ``(batch, time, d_model)``.
        """
        B, T, V, D = x.shape

        # --- per-variable GRN transforms ---
        # transformed[i]: (B, T, D)
        transformed_list: list[Tensor] = []
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint

            for i in range(self.n_vars):
                transformed_list.append(
                    checkpoint(self.var_grns[i], x[:, :, i, :], use_reentrant=False)
                )
        else:
            for i in range(self.n_vars):
                transformed_list.append(self.var_grns[i](x[:, :, i, :]))
        # (B, T, V, D)
        transformed = torch.stack(transformed_list, dim=2)

        # --- variable-selection weights ---
        # Flatten variables for the weight GRN: (B, T, V*D)
        flat = x.reshape(B, T, V * D)

        # Expand context to match time dimension if provided
        ctx = None
        if context is not None:
            # context: (B, context_dim) → (B, T, context_dim)
            ctx = context.unsqueeze(1).expand(-1, T, -1)

        # (B, T, n_vars)
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint

            weights = checkpoint(self.weight_grn, flat, ctx, use_reentrant=False)
        else:
            weights = self.weight_grn(flat, context=ctx)
        weights = F.softmax(weights, dim=-1)  # (B, T, V)

        # --- weighted sum ---
        # weights: (B, T, V, 1),  transformed: (B, T, V, D)
        out = (transformed * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)
        return out


# 3. Temporal Self-Attention
class TemporalSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional causal masking.

    After the attention computation a residual connection and Layer
    Normalization are applied.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    n_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout applied to attention weights (default 0.1).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # pylint: disable=invalid-name
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, time, d_model)``.
        mask : Tensor or None
            Optional boolean mask of shape ``(1, 1, time, time)`` where
            ``True`` indicates positions to **mask out** (set to ``-inf``).

        Returns
        -------
        output : Tensor
            Shape ``(batch, time, d_model)``.
        attn_weights : Tensor
            Shape ``(batch, n_heads, time, time)``.
        """
        B, T, _ = x.shape
        residual = x

        # --- Q, K, V projections ---
        Q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (B, n_heads, T, d_k)

        # --- scaled dot-product attention ---
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)  # (B, n_heads, T, d_k)

        # --- concatenate heads ---
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(attn_out)

        # --- residual + LayerNorm ---
        out = self.layer_norm(out + residual)

        return out, attn_weights


# 4. Static Encoder
class StaticEncoder(nn.Module):
    """Encodes static (time-invariant) features into four context vectors.

    Each context vector is produced by a dedicated GRN and serves a
    different role downstream:

    * **c_s** — context for Variable Selection Networks
    * **c_e** — context for static enrichment of temporal features
    * **c_h** — initial hidden state for the LSTM encoder / decoder
    * **c_c** — initial cell state for the LSTM encoder / decoder

    Parameters
    ----------
    n_static : int
        Number of static input features.
    d_model : int
        Model dimensionality.
    dropout : float, optional
        Dropout used inside GRNs (default 0.1).
    """

    def __init__(
        self,
        n_static: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Use proper embeddings for the 3 categorical static features
        # sector_id: 0..29, board_id: 0..9, cap_bin_id: 0..4
        self.sector_embed = nn.Embedding(30, d_model)
        self.board_embed = nn.Embedding(10, d_model)
        self.cap_embed = nn.Embedding(5, d_model)

        # Projection to fuse them
        self.fuse_proj = nn.Linear(d_model * 3, d_model)

        self.grn_cs = GRN(d_model, d_model, d_model, dropout)
        self.grn_ce = GRN(d_model, d_model, d_model, dropout)
        self.grn_ch = GRN(d_model, d_model, d_model, dropout)
        self.grn_cc = GRN(d_model, d_model, d_model, dropout)

    def forward(self, static_x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        static_x : Tensor
            Static features of shape ``(batch, n_static)``.

        Returns
        -------
        c_s : Tensor
            VSN context, shape ``(batch, d_model)``.
        c_e : Tensor
            Static-enrichment context, shape ``(batch, d_model)``.
        c_h : Tensor
            LSTM initial hidden state, shape ``(batch, d_model)``.
        c_c : Tensor
            LSTM initial cell state, shape ``(batch, d_model)``.
        """
        # static_x shape: (B, 3) where columns are sector_id, board_id, cap_bin_id
        sector_ids = static_x[:, 0].long()
        board_ids = static_x[:, 1].long()
        cap_ids = static_x[:, 2].long()

        sect_emb = self.sector_embed(sector_ids)  # (B, d_model)
        board_emb = self.board_embed(board_ids)  # (B, d_model)
        cap_emb = self.cap_embed(cap_ids)  # (B, d_model)

        # Concatenate and project
        fused = torch.cat([sect_emb, board_emb, cap_emb], dim=-1)  # (B, d_model * 3)
        embedded = self.fuse_proj(fused)  # (B, d_model)

        c_s = self.grn_cs(embedded)
        c_e = self.grn_ce(embedded)
        c_h = self.grn_ch(embedded)
        c_c = self.grn_cc(embedded)

        return c_s, c_e, c_h, c_c


# 5. TFT (Temporal Fusion Transformer – full model)
class TFT(nn.Module):
    # pylint: disable=too-many-instance-attributes
    """Temporal Fusion Transformer for multi-horizon forecasting.

    Implements the complete TFT pipeline:

    1. Static encoding → four context vectors.
    2. Per-variable linear projections for past / future inputs.
    3. Variable selection (VSN) for past and future.
    4. LSTM encoder (past) and decoder (future).
    5. Static enrichment via a GRN conditioned on *c_e*.
    6. Temporal multi-head self-attention.
    7. Position-wise feed-forward (post-attention GRN).
    8. Final output projection to scalar forecasts.

    Parameters
    ----------
    n_static : int
        Number of static features.
    n_past : int
        Number of past (known + observed) time-varying features.
    n_future : int
        Number of future (known) time-varying features.
    d_model : int, optional
        Internal model dimensionality (default 64).
    n_heads : int, optional
        Number of attention heads (default 4).
    n_lstm_layers : int, optional
        Number of stacked LSTM layers (default 2).
    dropout : float, optional
        Global dropout rate (default 0.1).
    lookback : int, optional
        Number of historical time steps (default 60).
    horizon : int, optional
        Number of future time steps to forecast (default 1).
    """

    def __init__(
        self,
        n_static: int,
        n_past: int,
        n_future: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        lookback: int = 60,
        horizon: int = 1,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.n_static = n_static
        self.n_past = n_past
        self.n_future = n_future
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_lstm_layers = n_lstm_layers
        self.lookback = lookback
        self.horizon = horizon

        # --- static encoder ---
        self.static_encoder = StaticEncoder(n_static, d_model, dropout)

        # --- per-variable input projections (vectorized parameters) ---
        self.past_var_w = nn.Parameter(torch.Tensor(n_past, 1, d_model))
        self.past_var_b = nn.Parameter(torch.Tensor(n_past, d_model))
        self.future_var_w = nn.Parameter(torch.Tensor(n_future, 1, d_model))
        self.future_var_b = nn.Parameter(torch.Tensor(n_future, d_model))

        # Kaiming uniform init matching standard PyTorch linear layer
        stdv_past = 1.0 / math.sqrt(1)
        nn.init.uniform_(self.past_var_w, -stdv_past, stdv_past)
        nn.init.zeros_(self.past_var_b)
        stdv_future = 1.0 / math.sqrt(1)
        nn.init.uniform_(self.future_var_w, -stdv_future, stdv_future)
        nn.init.zeros_(self.future_var_b)

        # --- variable selection networks ---
        self.vsn_past = VSN(
            n_past, d_model, dropout, context_dim=d_model, use_checkpoint=use_checkpoint
        )
        self.vsn_future = VSN(
            n_future,
            d_model,
            dropout,
            context_dim=d_model,
            use_checkpoint=use_checkpoint,
        )

        # --- LSTM encoder / decoder ---
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # --- static enrichment ---
        self.enrichment_grn = GRN(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
            context_dim=d_model,
        )

        # --- temporal self-attention ---
        self.temporal_attn = TemporalSelfAttention(d_model, n_heads, dropout)

        # --- post-attention GRN ---
        self.post_attn_grn = GRN(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
        )

        # --- output layers ---
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model >> 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model >> 1, 1),
        )

    # ----- helper: project variables -----
    def _project_variables(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """Project each variable independently to ``d_model`` using batched matrix multiplication.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, V)`` where V is the number of variables.
        w : Tensor
            Weight parameter of shape ``(V, 1, d_model)``.
        b : Tensor
            Bias parameter of shape ``(V, d_model)``.

        Returns
        -------
        Tensor
            Shape ``(B, T, V, d_model)``.
        """
        # x shape: (B, T, V)
        # w shape: (V, 1, D)
        # b shape: (V, D)
        # Permute x to (V, B, T, 1)
        x_in = x.permute(2, 0, 1).unsqueeze(-1)
        # Batched matmul: (V, B, T, 1) x (V, 1, 1, D) -> (V, B, T, D)
        out = torch.matmul(x_in, w.unsqueeze(1)) + b.unsqueeze(1).unsqueeze(1)
        # Permute back to (B, T, V, D)
        return out.permute(1, 2, 0, 3).contiguous()

    # ----- helper: prepare LSTM initial states -----
    def _init_lstm_states(self, c_h: Tensor, c_c: Tensor) -> tuple[Tensor, Tensor]:
        """Expand context vectors into multi-layer LSTM states.

        Parameters
        ----------
        c_h, c_c : Tensor
            Shape ``(B, d_model)``.

        Returns
        -------
        h0, c0 : Tensor
            Shape ``(n_lstm_layers, B, d_model)``.
        """
        # (B, d_model) → (1, B, d_model) → (n_layers, B, d_model)
        h0 = c_h.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()
        c0 = c_c.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()
        return h0, c0

    def forward(self, static_x: Tensor, past_x: Tensor, future_x: Tensor) -> Tensor:
        # pylint: disable=invalid-name,too-many-locals
        """Full TFT forward pass.

        Parameters
        ----------
        static_x : Tensor
            Static features, shape ``(B, n_static)``.
        past_x : Tensor
            Past time-varying features, shape ``(B, lookback, n_past)``.
        future_x : Tensor
            Future known features, shape ``(B, horizon, n_future)``.

        Returns
        -------
        Tensor
            Forecast of shape ``(B, horizon)``.
        """
        # 1. Static encoding
        c_s, c_e, c_h, c_c = self.static_encoder(static_x)

        # 2. Per-variable projections
        past_proj = self._project_variables(past_x, self.past_var_w, self.past_var_b)
        future_proj = self._project_variables(
            future_x, self.future_var_w, self.future_var_b
        )

        # 3. Variable selection
        past_selected = self.vsn_past(past_proj, context=c_s)  # (B, T_p, D)
        future_selected = self.vsn_future(future_proj, context=c_s)  # (B, T_f, D)

        # 4. LSTM encoder (past)
        h0, c0 = self._init_lstm_states(c_h, c_c)
        enc_out, (h_n, c_n) = self.lstm_encoder(past_selected, (h0, c0))

        # 5. LSTM decoder (future), initialised with encoder final states
        dec_out, _ = self.lstm_decoder(future_selected, (h_n, c_n))

        # 6. Concatenate encoder + decoder along the time axis
        temporal = torch.cat([enc_out, dec_out], dim=1)  # (B, T_p+T_f, D)

        # 7. Static enrichment — expand c_e over time
        T_total = temporal.size(1)
        c_e_expanded = c_e.unsqueeze(1).expand(-1, T_total, -1)
        enriched = self.enrichment_grn(temporal, context=c_e_expanded)

        # 8. Temporal self-attention (with causal mask)
        causal_mask = (
            torch.triu(
                torch.ones(T_total, T_total, device=temporal.device, dtype=torch.bool),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, T, T)
        attn_out, _ = self.temporal_attn(enriched, mask=causal_mask)

        # 9. Post-attention GRN
        post_attn = self.post_attn_grn(attn_out)

        # 10. Take only the last `horizon` time steps
        out = post_attn[:, -self.horizon :, :]  # (B, horizon, D)

        # 11. Output projection → (B, horizon, 1) → (B, horizon)
        out = self.output_fc(out).squeeze(-1)

        return out


# 6. TFT_NoVSN (Ablation: no variable selection)
class TFT_NoVSN(nn.Module):
    # pylint: disable=too-many-instance-attributes
    """TFT ablation variant that replaces VSN with simple variable averaging.

    Instead of learned variable-selection weights, the per-variable
    projections are combined by taking their mean along the variable
    dimension.  All other components remain identical to :class:`TFT`.

    Parameters
    ----------
    (same as TFT)
    """

    def __init__(
        self,
        n_static: int,
        n_past: int,
        n_future: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        lookback: int = 60,
        horizon: int = 1,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.n_static = n_static
        self.n_past = n_past
        self.n_future = n_future
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_lstm_layers = n_lstm_layers
        self.lookback = lookback
        self.horizon = horizon

        # --- static encoder ---
        self.static_encoder = StaticEncoder(n_static, d_model, dropout)

        # --- per-variable input projections (vectorized parameters) ---
        self.past_var_w = nn.Parameter(torch.Tensor(n_past, 1, d_model))
        self.past_var_b = nn.Parameter(torch.Tensor(n_past, d_model))
        self.future_var_w = nn.Parameter(torch.Tensor(n_future, 1, d_model))
        self.future_var_b = nn.Parameter(torch.Tensor(n_future, d_model))

        # Kaiming uniform init matching standard PyTorch linear layer
        stdv_past = 1.0 / math.sqrt(1)
        nn.init.uniform_(self.past_var_w, -stdv_past, stdv_past)
        nn.init.zeros_(self.past_var_b)
        stdv_future = 1.0 / math.sqrt(1)
        nn.init.uniform_(self.future_var_w, -stdv_future, stdv_future)
        nn.init.zeros_(self.future_var_b)

        # NO VSN — simple averaging is used instead

        # --- LSTM encoder / decoder ---
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # --- static enrichment ---
        self.enrichment_grn = GRN(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
            context_dim=d_model,
        )

        # --- temporal self-attention ---
        self.temporal_attn = TemporalSelfAttention(d_model, n_heads, dropout)

        # --- post-attention GRN ---
        self.post_attn_grn = GRN(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
        )

        # --- output layers ---
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model >> 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model >> 1, 1),
        )

    def _project_variables(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """Project each variable independently to ``d_model``.

        Returns shape ``(B, T, V, d_model)``.
        """
        x_in = x.permute(2, 0, 1).unsqueeze(-1)
        out = torch.matmul(x_in, w.unsqueeze(1)) + b.unsqueeze(1).unsqueeze(1)
        return out.permute(1, 2, 0, 3).contiguous()

    def _init_lstm_states(self, c_h: Tensor, c_c: Tensor) -> tuple[Tensor, Tensor]:
        """Expand context vectors into multi-layer LSTM states."""
        h0 = c_h.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()
        c0 = c_c.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()
        return h0, c0

    def forward(self, static_x: Tensor, past_x: Tensor, future_x: Tensor) -> Tensor:
        # pylint: disable=invalid-name,too-many-locals
        """Forward pass — identical to TFT except VSN is replaced by mean.

        Parameters
        ----------
        static_x : Tensor - ``(B, n_static)``
        past_x   : Tensor - ``(B, lookback, n_past)``
        future_x : Tensor - ``(B, horizon, n_future)``

        Returns
        -------
        Tensor - ``(B, horizon)``
        """
        # 1. Static encoding
        c_s, c_e, c_h, c_c = self.static_encoder(static_x)

        # 2. Per-variable projections
        past_proj = self._project_variables(past_x, self.past_var_w, self.past_var_b)
        future_proj = self._project_variables(
            future_x, self.future_var_w, self.future_var_b
        )

        # 3. Simple averaging instead of VSN
        past_selected = past_proj.mean(dim=2)  # (B, T_p, D)
        future_selected = future_proj.mean(dim=2)  # (B, T_f, D)

        # 4. LSTM encoder
        h0, c0 = self._init_lstm_states(c_h, c_c)
        enc_out, (h_n, c_n) = self.lstm_encoder(past_selected, (h0, c0))

        # 5. LSTM decoder
        dec_out, _ = self.lstm_decoder(future_selected, (h_n, c_n))

        # 6. Concatenate
        temporal = torch.cat([enc_out, dec_out], dim=1)

        # 7. Static enrichment
        T_total = temporal.size(1)
        c_e_expanded = c_e.unsqueeze(1).expand(-1, T_total, -1)
        enriched = self.enrichment_grn(temporal, context=c_e_expanded)

        # 8. Temporal self-attention
        causal_mask = (
            torch.triu(
                torch.ones(T_total, T_total, device=temporal.device, dtype=torch.bool),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        attn_out, _ = self.temporal_attn(enriched, mask=causal_mask)

        # 9. Post-attention GRN
        post_attn = self.post_attn_grn(attn_out)

        # 10. Take horizon steps
        out = post_attn[:, -self.horizon :, :]

        # 11. Output
        out = self.output_fc(out).squeeze(-1)
        return out


# 7. TFT_NoAttention (Ablation: no self-attention)
class TFT_NoAttention(nn.Module):
    # pylint: disable=too-many-instance-attributes
    """TFT ablation variant that skips the temporal self-attention block.

    The statically-enriched temporal representation is fed directly into
    the post-attention GRN, bypassing multi-head attention entirely.
    All other components remain identical to :class:`TFT`.

    Parameters
    ----------
    (same as TFT)
    """

    def __init__(
        self,
        n_static: int,
        n_past: int,
        n_future: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        lookback: int = 60,
        horizon: int = 1,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.n_static = n_static
        self.n_past = n_past
        self.n_future = n_future
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_lstm_layers = n_lstm_layers
        self.lookback = lookback
        self.horizon = horizon

        # --- static encoder ---
        self.static_encoder = StaticEncoder(n_static, d_model, dropout)

        # --- per-variable input projections (vectorized parameters) ---
        self.past_var_w = nn.Parameter(torch.Tensor(n_past, 1, d_model))
        self.past_var_b = nn.Parameter(torch.Tensor(n_past, d_model))
        self.future_var_w = nn.Parameter(torch.Tensor(n_future, 1, d_model))
        self.future_var_b = nn.Parameter(torch.Tensor(n_future, d_model))

        # Kaiming uniform init matching standard PyTorch linear layer
        stdv_past = 1.0 / math.sqrt(1)
        nn.init.uniform_(self.past_var_w, -stdv_past, stdv_past)
        nn.init.zeros_(self.past_var_b)
        stdv_future = 1.0 / math.sqrt(1)
        nn.init.uniform_(self.future_var_w, -stdv_future, stdv_future)
        nn.init.zeros_(self.future_var_b)

        # --- variable selection networks ---
        self.vsn_past = VSN(
            n_past, d_model, dropout, context_dim=d_model, use_checkpoint=use_checkpoint
        )
        self.vsn_future = VSN(
            n_future,
            d_model,
            dropout,
            context_dim=d_model,
            use_checkpoint=use_checkpoint,
        )

        # --- LSTM encoder / decoder ---
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # --- static enrichment ---
        self.enrichment_grn = GRN(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
            context_dim=d_model,
        )

        # NO temporal self-attention

        # --- post-attention GRN ---
        self.post_attn_grn = GRN(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
        )

        # --- output layers ---
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model >> 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model >> 1, 1),
        )

    # ----- helpers (same as TFT) -----
    def _project_variables(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """Project each variable independently to ``d_model``.

        Returns shape ``(B, T, V, d_model)``.
        """
        x_in = x.permute(2, 0, 1).unsqueeze(-1)
        out = torch.matmul(x_in, w.unsqueeze(1)) + b.unsqueeze(1).unsqueeze(1)
        return out.permute(1, 2, 0, 3).contiguous()

    def _init_lstm_states(self, c_h: Tensor, c_c: Tensor) -> tuple[Tensor, Tensor]:
        """Expand context vectors into multi-layer LSTM states."""
        h0 = c_h.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()
        c0 = c_c.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()
        return h0, c0

    def forward(self, static_x: Tensor, past_x: Tensor, future_x: Tensor) -> Tensor:
        # pylint: disable=invalid-name,too-many-locals
        """Forward pass — identical to TFT but without the attention block.

        Parameters
        ----------
        static_x : Tensor - ``(B, n_static)``
        past_x   : Tensor - ``(B, lookback, n_past)``
        future_x : Tensor - ``(B, horizon, n_future)``

        Returns
        -------
        Tensor - ``(B, horizon)``
        """
        # 1. Static encoding
        c_s, c_e, c_h, c_c = self.static_encoder(static_x)

        # 2. Per-variable projections
        past_proj = self._project_variables(past_x, self.past_var_w, self.past_var_b)
        future_proj = self._project_variables(
            future_x, self.future_var_w, self.future_var_b
        )

        # 3. Variable selection
        past_selected = self.vsn_past(past_proj, context=c_s)
        future_selected = self.vsn_future(future_proj, context=c_s)

        # 4. LSTM encoder
        h0, c0 = self._init_lstm_states(c_h, c_c)
        enc_out, (h_n, c_n) = self.lstm_encoder(past_selected, (h0, c0))

        # 5. LSTM decoder
        dec_out, _ = self.lstm_decoder(future_selected, (h_n, c_n))

        # 6. Concatenate
        temporal = torch.cat([enc_out, dec_out], dim=1)

        # 7. Static enrichment
        T_total = temporal.size(1)
        c_e_expanded = c_e.unsqueeze(1).expand(-1, T_total, -1)
        enriched = self.enrichment_grn(temporal, context=c_e_expanded)

        # 8. SKIP attention — pass enriched directly to post-attention GRN

        # 9. Post-attention GRN
        post_attn = self.post_attn_grn(enriched)

        # 10. Take horizon steps
        out = post_attn[:, -self.horizon :, :]

        # 11. Output
        out = self.output_fc(out).squeeze(-1)
        return out

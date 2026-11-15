"""Attention pooling de type ABMIL pour agregation slide-level."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ABMILAttention(nn.Module):
    """Implementation compacte d'un mecanisme d'attention ABMIL classique."""

    def __init__(self, input_dim: int = 384, attention_dim: int = 256) -> None:
        super().__init__()
        self.attention_v = nn.Linear(input_dim, attention_dim)
        self.attention_u = nn.Linear(input_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Agrege les tokens de patchs en un unique vecteur slide-level.
        
        Si `mask` est fourni (booleen, True = masquer), les logits correspondants
        sont forces a -1e9 avant le softmax.
        """

        v = torch.tanh(self.attention_v(tokens))
        u = torch.sigmoid(self.attention_u(tokens))
        attention_logits = self.attention_w(v * u).squeeze(-1)

        if mask is not None:
            # -1e9 garantit que le softmax associe un poids de 0 a ces tokens
            attention_logits = attention_logits.masked_fill(mask, -1e9)
        attention_weights = torch.softmax(attention_logits, dim=0)
        slide_embedding = torch.sum(attention_weights.unsqueeze(-1) * tokens, dim=0)

        if return_attention:
            return slide_embedding, attention_weights
        return slide_embedding


class TransMILAttention(nn.Module):
    """Implementation de TransMIL: Self-Attention (Transformer) suivi par ABMIL pooling."""

    def __init__(
        self,
        input_dim: int = 384,
        attention_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert input_dim % n_heads == 0, f"input_dim {input_dim} must be divisible by n_heads {n_heads}"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Standard ABMIL to pool the transformer outputs into 1 slide_embedding
        self.pooling = ABMILAttention(input_dim=input_dim, attention_dim=attention_dim)

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Agrege les tokens via un Transformer puis ABMIL pooling.

        Attention: `tokens` est de forme [N, D] (pas de dimension batch explicite ici
        car on traite une lame a la fois).
        """
        # Transformer expects [Batch, Seq_Len, Dim]
        x = tokens.unsqueeze(0)

        # PyTorch Transformer src_key_padding_mask expects [Batch, Seq_Len].
        # True means 'ignore this token', aligning perfectly with our artifact_mask.
        if mask is not None:
            pad_mask = mask.unsqueeze(0)
            x = self.transformer(x, src_key_padding_mask=pad_mask)
        else:
            x = self.transformer(x)

        # Remove batch dim: [N, D]
        x = x.squeeze(0)

        # Apply ABMIL pooling on the context-aware patches
        return self.pooling(x, mask=mask, return_attention=return_attention)

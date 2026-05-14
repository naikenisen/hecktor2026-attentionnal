import torch
import torch.nn as nn

# Fusionne les tokens [CLS, clinique, TN] avec le bottleneck spatial via cross-attention
class CrossAttentionFusion(nn.Module):

    # Initialise la projection K/V, l'attention multi-têtes, les norms et le FFN
    def __init__(self, bottleneck_channels: int, d_model: int = 256, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        # Projette les features bottleneck (C) vers l'espace d_model pour former K et V
        self.kv_proj = nn.Linear(bottleneck_channels, d_model)
        # Attention multi-têtes batch_first : Q=(B,3,d), K=V=(B,N,d) → (B,3,d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Normalisation appliquée aux queries avant l'attention
        self.norm_q = nn.LayerNorm(d_model)
        # Normalisation appliquée aux keys/values avant l'attention
        self.norm_kv = nn.LayerNorm(d_model)
        # FFN post-attention avec GELU et dropout
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        # Normalisation appliquée avant le FFN
        self.norm_ffn = nn.LayerNorm(d_model)

    # Renvoie uniquement le token CLS enrichi (B, d_model) après attention et FFN
    def forward(self, queries: torch.Tensor, bottleneck: torch.Tensor) -> torch.Tensor:
        # Nombre de voxels dans le bottleneck après aplatissement
        B, C = bottleneck.shape[:2]
        # Aplatit et projette le bottleneck en séquence (B, N, d_model)
        kv = bottleneck.flatten(2).permute(0, 2, 1)
        kv = self.kv_proj(kv)
        kv = self.norm_kv(kv)
        # Normalise les queries avant l'attention
        q = self.norm_q(queries)
        # Calcule l'attention croisée Q → (K, V) sans retourner les poids
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        # Connexion résiduelle post-attention
        x = queries + attn_out
        # Connexion résiduelle post-FFN
        x = x + self.ffn(self.norm_ffn(x))
        # Retourne uniquement le premier token (CLS)
        return x[:, 0]

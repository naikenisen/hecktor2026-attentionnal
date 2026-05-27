import torch
import torch.nn as nn

from src.heads import TNHead, SurvivalHead
from src.clinical_encoder import ClinicalMLP
from src.cross_attention import CrossAttentionFusion


# Branche clinique découplée : consomme un bottleneck pré-calculé (figé) + le vecteur
# clinique, et produit les logits T/N et survie. C'est exactement le forward de
# MultitaskModel à partir du bottleneck, sans le backbone — pour l'entraînement phase 2.
class ClinicalModel(nn.Module):

    # Instancie les têtes T/N, la fusion cross-attention et la tête survie
    def __init__(self, config):
        super().__init__()
        # Nombre de canaux du bottleneck (768 pour feature_size=48)
        C = config.bottleneck_channels
        # Tête de classification du staging T
        self.t_head = TNHead(C, hidden_dim=config.hidden_tn,
                             num_classes=config.num_t_classes)
        # Tête de classification du staging N
        self.n_head = TNHead(C, hidden_dim=config.hidden_tn,
                             num_classes=config.num_n_classes)
        # Projette [t_feat, n_feat] vers d_model pour former le token TN
        self.proj_tn = nn.Linear(2 * config.hidden_tn, config.d_model)
        # MLP clinique encodant les 7 features en token (B, 1, d_model)
        self.clin_mlp = ClinicalMLP(n_features=config.n_clinical_features,
                                    hidden_dim=64, d_model=config.d_model)
        # Token CLS appris, query principale de la cross-attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        # Fusion cross-attention entre les tokens et le bottleneck spatial
        self.cross_attn = CrossAttentionFusion(
            bottleneck_channels=C,
            d_model=config.d_model,
            n_heads=config.n_heads,
        )
        # Tête de survie discrète sur T intervalles
        self.surv_head = SurvivalHead(
            d_model=config.d_model,
            hidden_dim=config.surv_hidden,
            n_time_bins=config.n_time_bins,
        )

    # Forward depuis le bottleneck : renvoie un dict avec t/n/surv logits
    def forward(self, bottleneck: torch.Tensor, clinical: torch.Tensor) -> dict:
        # Features riches et logits T
        t_feat, t_logits = self.t_head(bottleneck)
        # Features riches et logits N
        n_feat, n_logits = self.n_head(bottleneck)

        # Token TN issu de la concaténation des features T et N
        token_tn = self.proj_tn(torch.cat([t_feat, n_feat], dim=1)).unsqueeze(1)
        # Token clinique
        token_clin = self.clin_mlp(clinical)

        # Taille du batch
        B = bottleneck.size(0)
        # CLS token expandé au batch
        cls = self.cls_token.expand(B, -1, -1).contiguous()
        # Queries concaténées : CLS, clinique, TN
        Q = torch.cat([cls, token_clin, token_tn], dim=1)

        # Token CLS enrichi par cross-attention avec le bottleneck spatial
        cls_out = self.cross_attn(Q, bottleneck)
        # Logits de survie bruts
        surv_logits = self.surv_head(cls_out)

        return {
            "t_logits":    t_logits,
            "n_logits":    n_logits,
            "surv_logits": surv_logits,
        }

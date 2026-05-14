import torch
import torch.nn as nn

from src.swinunetr import SwinUNETRConfig, SwinUNETRMultitask
from src.heads import TNHead, SurvivalHead
from src.clinical_encoder import ClinicalMLP
from src.cross_attention import CrossAttentionFusion

# Modèle multitâche end-to-end : segmentation + staging T/N + survie discrète
class MultitaskModel(nn.Module):

    # Instancie et connecte tous les sous-modules de la pipeline
    def __init__(self, config):
        super().__init__()
        # Référence à la configuration globale de la pipeline
        self.config = config

        # Configuration du backbone SwinUNETR à partir de la config globale
        sw_cfg = SwinUNETRConfig(
            input_channels=config.input_channels,
            num_classes=config.num_seg_classes,
            spatial_size=config.spatial_size,
            feature_size=config.feature_size,
            use_checkpoint=config.use_checkpoint,
            pretrained_path=config.pretrained_path,
        )
        # Backbone SwinUNETR qui produit (seg_mask, bottleneck)
        self.backbone = SwinUNETRMultitask(sw_cfg)

        # Nombre de canaux du bottleneck (768 pour feature_size=48)
        C_bottleneck = config.bottleneck_channels
        # Tête de classification pour le staging T (4 classes)
        self.t_head = TNHead(C_bottleneck, hidden_dim=config.hidden_tn,
                             num_classes=config.num_t_classes)
        # Tête de classification pour le staging N (4 classes)
        self.n_head = TNHead(C_bottleneck, hidden_dim=config.hidden_tn,
                             num_classes=config.num_n_classes)

        # Projette la concaténation [t_feat, n_feat] (512) vers d_model pour former token_tn
        self.proj_tn = nn.Linear(2 * config.hidden_tn, config.d_model)
        # MLP clinique qui encode les 7 features cliniques en token (B, 1, d_model)
        self.clin_mlp = ClinicalMLP(n_features=config.n_clinical_features,
                                    hidden_dim=64, d_model=config.d_model)
        # Token CLS appris, utilisé comme query principale dans la cross-attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Module de fusion cross-attention entre les tokens et le bottleneck spatial
        self.cross_attn = CrossAttentionFusion(
            bottleneck_channels=C_bottleneck,
            d_model=config.d_model,
            n_heads=config.n_heads,
        )
        # Tête de survie discrète qui prédit les logits sur T intervalles
        self.surv_head = SurvivalHead(
            d_model=config.d_model,
            hidden_dim=config.surv_hidden,
            n_time_bins=config.n_time_bins,
        )

    # Exécute le forward complet et renvoie un dict avec les 4 sorties
    def forward(self, ct_pet: torch.Tensor, clinical: torch.Tensor) -> dict:
        # Segmentation et extraction du bottleneck par le backbone
        seg_mask, bottleneck = self.backbone(ct_pet)

        # Features riches et logits de staging T
        t_feat, t_logits = self.t_head(bottleneck)
        # Features riches et logits de staging N
        n_feat, n_logits = self.n_head(bottleneck)

        # Token TN construit depuis les features T et N concaténées
        token_tn = self.proj_tn(torch.cat([t_feat, n_feat], dim=1)).unsqueeze(1)
        # Token clinique issu du MLP sur les 7 variables cliniques
        token_clin = self.clin_mlp(clinical)

        # Taille du batch courant
        B = ct_pet.size(0)
        # CLS token expandé à la taille du batch
        cls = self.cls_token.expand(B, -1, -1).contiguous()

        # Concaténation des 3 queries : CLS, token clinique, token TN
        Q = torch.cat([cls, token_clin, token_tn], dim=1)

        # Token CLS enrichi par cross-attention avec le bottleneck spatial
        cls_out = self.cross_attn(Q, bottleneck)

        # Logits de survie bruts sur T intervalles (softmax appliqué à l'inférence uniquement)
        surv_logits = self.surv_head(cls_out)

        return {
            "seg_mask":    seg_mask,
            "t_logits":    t_logits,
            "n_logits":    n_logits,
            "surv_logits": surv_logits,
        }

    # Sauvegarde le modèle et l'état de l'optimiseur dans un fichier .pth
    def save_checkpoint(self, path: str, epoch: int, optimizer_state=None, **kwargs):
        # Dictionnaire du checkpoint avec les poids et métadonnées
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            **kwargs,
        }
        if optimizer_state is not None:
            ckpt["optimizer_state_dict"] = optimizer_state
        torch.save(ckpt, path)

    # Charge les poids depuis un checkpoint et retourne le dictionnaire complet
    def load_checkpoint(self, path: str, device: str = "cpu") -> dict:
        ckpt = torch.load(path, map_location=device)
        self.load_state_dict(ckpt["model_state_dict"])
        return ckpt

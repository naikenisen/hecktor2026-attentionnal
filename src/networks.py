import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


# Sous-classe MONAI SwinUNETR qui expose le bottleneck en plus du masque de segmentation
class SwinUNETRBackbone(nn.Module):

    # Instancie le SwinUNETR MONAI et charge les poids SSL si disponibles
    def __init__(self, input_channels: int, num_classes: int, feature_size: int,
                 use_checkpoint: bool, pretrained_path: str | None = None):
        super().__init__()

        # Instance SwinUNETR MONAI standard (sera appelé manuellement dans forward)
        self.swinunetr = SwinUNETR(
            in_channels=input_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )


        weights = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        self.swinunetr.load_from(weights=weights)
        print(f"[SwinUNETRBackbone] SSL weights loaded from '{pretrained_path}'.")

    # Retourne (seg_logits, bottleneck) en reproduisant manuellement le forward MONAI
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Raccourci vers l'instance SwinUNETR interne
        net = self.swinunetr
        # Liste des feature maps hiérarchiques [stage0..stage3, bottleneck]
        hidden_states_out = net.swinViT(x, net.normalize)
        # Feature map la plus profonde (downscale ×32, C=feature_size×16)
        bottleneck = hidden_states_out[4]

        # Encodeur skip-connection niveau 0 (résolution originale)
        enc0 = net.encoder1(x)
        # Encodeur skip-connection niveau 1
        enc1 = net.encoder2(hidden_states_out[0])
        # Encodeur skip-connection niveau 2
        enc2 = net.encoder3(hidden_states_out[1])
        # Encodeur skip-connection niveau 3
        enc3 = net.encoder4(hidden_states_out[2])
        # Encodeur bottleneck (niveau 4)
        dec4 = net.encoder10(hidden_states_out[4])
        # Décodeur niveau 3
        dec3 = net.decoder5(dec4, hidden_states_out[3])
        # Décodeur niveau 2
        dec2 = net.decoder4(dec3, enc3)
        # Décodeur niveau 1
        dec1 = net.decoder3(dec2, enc2)
        # Décodeur niveau 0
        dec0 = net.decoder2(dec1, enc1)
        # Décodeur final avant la couche de sortie
        out = net.decoder1(dec0, enc0)
        # Logits de segmentation (B, num_classes, D, H, W)
        seg_logits = net.out(out)

        return seg_logits, bottleneck


# Tête de classification pour le staging T ou N à partir du bottleneck
class TNHead(nn.Module):

    # Initialise le GAP, la couche cachée et le classifieur final
    def __init__(self, in_channels: int, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        # Réduit le volume spatial (B, C, D', H', W') à (B, C) par moyenne globale
        self.gap = nn.AdaptiveAvgPool3d(1)
        # Projette les features vers la dimension cachée avec activation et dropout
        self.feat = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        # Projette la feature cachée vers les logits de classification
        self.classifier = nn.Linear(hidden_dim, num_classes)

    # Renvoie la feature riche (B, hidden_dim) et les logits (B, num_classes)
    def forward(self, bottleneck: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Aplati le bottleneck en (B, C) via GAP
        x = self.gap(bottleneck).flatten(1)
        # Vecteur de feature intermédiaire utilisé pour la fusion cross-attention
        feat = self.feat(x)
        # Logits bruts pour la classification T ou N
        logits = self.classifier(feat)
        return feat, logits


# Tête de survie discrète qui produit des logits sur T intervalles temporels
class SurvivalHead(nn.Module):

    # Initialise le MLP d_model → hidden → T bins
    def __init__(self, d_model: int, hidden_dim: int = 256, n_time_bins: int = 10):
        super().__init__()
        # Réseau fully-connected qui mappe le token CLS enrichi vers les logits de survie
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_time_bins),
        )

    # Renvoie les logits bruts (B, T) ; le softmax est appliqué uniquement à l'inférence
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Encode le vecteur clinique (B, 7) en un token (B, 1, d_model) pour la cross-attention
class ClinicalMLP(nn.Module):

    # Initialise le MLP 7 → hidden → d_model
    def __init__(self, n_features: int = 7, hidden_dim: int = 64, d_model: int = 256):
        super().__init__()
        # Réseau fully-connected qui projette les 7 features cliniques vers l'espace d_model
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, d_model),
        )

    # Renvoie le token clinique (B, 1, d_model) prêt pour la cross-attention
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).unsqueeze(1)


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

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRBackbone(nn.Module):

    def __init__(self, input_channels: int, num_classes: int, feature_size: int,
                 use_checkpoint: bool, pretrained_path: str | None = None):
        super().__init__()
        self.swinunetr = SwinUNETR(
            in_channels=input_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )
        weights = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        self.swinunetr.load_from(weights=weights)
        print(f"[SwinUNETRBackbone] SSL weights loaded from '{pretrained_path}'.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net = self.swinunetr
        hidden_states_out = net.swinViT(x, net.normalize)
        bottleneck = hidden_states_out[4]
        enc0 = net.encoder1(x)
        enc1 = net.encoder2(hidden_states_out[0])
        enc2 = net.encoder3(hidden_states_out[1])
        enc3 = net.encoder4(hidden_states_out[2])
        dec4 = net.encoder10(hidden_states_out[4])
        dec3 = net.decoder5(dec4, hidden_states_out[3])
        dec2 = net.decoder4(dec3, enc3)
        dec1 = net.decoder3(dec2, enc2)
        dec0 = net.decoder2(dec1, enc1)
        out = net.decoder1(dec0, enc0)
        seg_logits = net.out(out)
        return seg_logits, bottleneck

class TNHead(nn.Module):

    def __init__(self, in_channels: int, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.feat = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, bottleneck: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.gap(bottleneck).flatten(1)
        feat = self.feat(x)
        logits = self.classifier(feat)
        return feat, logits

class SurvivalHead(nn.Module):

    def __init__(self, d_model: int, hidden_dim: int = 256, n_time_bins: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_time_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ClinicalMLP(nn.Module):

    def __init__(self, n_features: int = 7, hidden_dim: int = 64, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).unsqueeze(1)

class CrossAttentionFusion(nn.Module):

    def __init__(self, bottleneck_channels: int, d_model: int = 256, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.kv_proj = nn.Linear(bottleneck_channels, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, bottleneck: torch.Tensor) -> torch.Tensor:
        B, C = bottleneck.shape[:2]
        kv = bottleneck.flatten(2).permute(0, 2, 1)
        kv = self.kv_proj(kv)
        kv = self.norm_kv(kv)
        q = self.norm_q(queries)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = queries + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x[:, 0]

class ClinicalModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        C = config.bottleneck_channels
        self.t_head = TNHead(C, hidden_dim=config.hidden_tn,
                             num_classes=config.num_t_classes)
        self.n_head = TNHead(C, hidden_dim=config.hidden_tn,
                             num_classes=config.num_n_classes)
        self.proj_tn = nn.Linear(2 * config.hidden_tn, config.d_model)
        self.clin_mlp = ClinicalMLP(n_features=config.n_clinical_features,
                                    hidden_dim=64, d_model=config.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.cross_attn = CrossAttentionFusion(
            bottleneck_channels=C,
            d_model=config.d_model,
            n_heads=config.n_heads,
        )
        self.surv_head = SurvivalHead(
            d_model=config.d_model,
            hidden_dim=config.surv_hidden,
            n_time_bins=config.n_time_bins,
        )

    def forward(self, bottleneck: torch.Tensor, clinical: torch.Tensor) -> dict:
        t_feat, t_logits = self.t_head(bottleneck)
        n_feat, n_logits = self.n_head(bottleneck)
        token_tn = self.proj_tn(torch.cat([t_feat, n_feat], dim=1)).unsqueeze(1)
        token_clin = self.clin_mlp(clinical)
        B = bottleneck.size(0)
        cls = self.cls_token.expand(B, -1, -1).contiguous()
        Q = torch.cat([cls, token_clin, token_tn], dim=1)
        cls_out = self.cross_attn(Q, bottleneck)
        surv_logits = self.surv_head(cls_out)
        return {
            "t_logits":    t_logits,
            "n_logits":    n_logits,
            "surv_logits": surv_logits,
        }

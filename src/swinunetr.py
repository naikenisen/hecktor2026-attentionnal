import os
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

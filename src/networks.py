import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRBackbone(nn.Module):
    """SwinUNETR-V2 de segmentation dont le forward expose aussi la carte de bottleneck
    profonde, réutilisée comme embedding figé (TEP/CT) par les têtes tabulaires."""

    def __init__(self, input_channels, num_classes, feature_size,
                 use_checkpoint, pretrained_path=None):
        super().__init__()
        self.swinunetr = SwinUNETR(
            in_channels=input_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            use_v2=True,  # variante SwinUNETR-V2 : blocs conv résiduels en tête de chaque étage
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

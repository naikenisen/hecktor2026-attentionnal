import torch
import torch.nn as nn

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

import torch
import torch.nn as nn

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

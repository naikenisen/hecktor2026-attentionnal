import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss

# Perte de segmentation combinant Dice et Focal Loss (via MONAI)
seg_loss = DiceFocalLoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False,
    reduction="mean",
)

# Perte cross-entropie pour la classification du staging T (−1 = label inconnu, ignoré)
t_loss = nn.CrossEntropyLoss(ignore_index=-1)
# Perte cross-entropie pour la classification du staging N (−1 = label inconnu, ignoré)
n_loss = nn.CrossEntropyLoss(ignore_index=-1)


# Perte DeepHit discrète : NLL sur les bins de survie + terme de ranking sur la CIF
class DeepHitDiscreteLoss(nn.Module):

    # Initialise les hyperparamètres alpha (poids ranking) et sigma (échelle ranking)
    def __init__(self, alpha: float = 0.2, sigma: float = 0.1):
        super().__init__()
        # Poids du terme de ranking par rapport au NLL
        self.alpha = alpha
        # Paramètre d'échelle de la pénalité exponentielle du ranking
        self.sigma = sigma

    # Convertit des temps continus en indices de bins discrets (0..T−1)
    @staticmethod
    def time_to_bin(times: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
        # Nombre de bins temporels
        T = bin_edges.numel() - 1
        # Indexation par bucketize puis clamp pour rester dans [0, T−1]
        idx = torch.bucketize(times, bin_edges[1:-1], right=False)
        return idx.clamp(0, T - 1).long()

    # Calcule la perte totale L_NLL + alpha * L_rank sur le batch
    def forward(self, logits: torch.Tensor, times: torch.Tensor,
                events: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
        # Périphérique courant
        device = logits.device
        # Taille du batch et nombre de bins
        B, T = logits.shape
        # Distribution de masse de probabilité sur les T bins (B, T)
        pmf = torch.softmax(logits, dim=-1)
        # Fonction d'incidence cumulée (CIF) : probabilité cumulée d'événement (B, T)
        cif = torch.cumsum(pmf, dim=-1)

        # Indice de bin pour chaque patient selon son temps de suivi (B,)
        k = self.time_to_bin(times, bin_edges)
        # Masque booléen des patients avec un événement observé
        events_b = events.bool()

        # Petite valeur pour éviter log(0)
        eps = 1e-8
        # Probabilité PMF au bin de l'événement pour chaque patient (B,)
        p_event = pmf.gather(1, k.unsqueeze(1)).squeeze(1)
        # Probabilité de survie au-delà du bin de censure : 1 − CIF[k] (B,)
        s_censor = 1.0 - cif.gather(1, k.unsqueeze(1)).squeeze(1)

        # NLL pour les patients avec événement
        nll_event = -torch.log(p_event[events_b].clamp_min(eps)).sum() \
                    if events_b.any() else torch.tensor(0.0, device=device)
        # NLL pour les patients censurés
        nll_censor = -torch.log(s_censor[~events_b].clamp_min(eps)).sum() \
                     if (~events_b).any() else torch.tensor(0.0, device=device)
        # NLL moyenne sur le batch
        l_nll = (nll_event + nll_censor) / max(B, 1)

        if events_b.any():
            # CIF de chaque patient évaluée à son propre bin k_i (B, 1)
            cif_at_ki = cif.gather(1, k.unsqueeze(1))
            # Expansion de k pour indexer la CIF de tous les patients j au bin k_i
            ki_expand = k.view(1, B).expand(B, B)
            # CIF du patient j évaluée au bin k_i de i — matrice (B_i, B_j)
            cif_j_at_ki = cif.unsqueeze(0).expand(B, B, T).gather(2, ki_expand.unsqueeze(-1)).squeeze(-1)
            # CIF du patient i évaluée à son propre bin, broadcastée à (B_i, B_j)
            cif_i_at_ki = cif_at_ki.expand(B, B)

            # Masque des paires où t_i < t_j (i précède j dans le temps)
            time_mat = times.view(B, 1) < times.view(1, B)
            # Paires valides : i a un événement et survient avant j
            valid = events_b.view(B, 1) & time_mat

            # Différence de CIF : positif si i est bien rangé devant j
            diff = cif_i_at_ki - cif_j_at_ki
            # Pénalité exponentielle pour les paires mal classées
            rank_loss = torch.exp(-diff / self.sigma)
            rank_loss = (rank_loss * valid.float()).sum() / (valid.float().sum() + eps)
        else:
            # Pas de paires valides dans ce batch
            rank_loss = torch.tensor(0.0, device=device)

        return l_nll + self.alpha * rank_loss


# Pondération des pertes multitâches par incertitude apprise (Kendall et al. 2018)
class UncertaintyWeightedLoss(nn.Module):

    # Initialise un log(σ) appris par tâche (σ=1 au démarrage)
    def __init__(self, n_tasks: int = 4, log_sigma_min: float = -3.0,
                 log_sigma_max: float = 3.0):
        super().__init__()
        # Paramètres appris log(σᵢ) — un par tâche — ajoutés à l'optimiseur
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))
        # Bornes de log(σ) : empêchent les poids 1/(2σ²) d'exploser (cause de NaN)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    # Combine les N pertes scalaires et retourne (perte_totale, poids_détachés)
    def forward(self, *task_losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(task_losses) == self.log_sigma.numel(), \
            f"Expected {self.log_sigma.numel()} losses, got {len(task_losses)}"

        # log(σ) borné pour garder les poids dans une plage numériquement stable
        log_sigma = self.log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
        # Variance de chaque tâche σᵢ²
        sigma_sq = torch.exp(2 * log_sigma)
        # Poids de chaque tâche : 1/(2σᵢ²)
        weights = 1.0 / (2.0 * sigma_sq)

        # Perte totale = Σ [ wᵢ · Lᵢ + log(σᵢ) ] selon la formule de Kendall
        total = sum(w * l + s for w, l, s in
                    zip(weights, task_losses, log_sigma))
        return total, weights.detach()

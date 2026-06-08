import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss

seg_loss = DiceFocalLoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False,
    reduction="mean",
    batch=True,  # agrège le terme Dice sur le batch : stabilise les structures petites/éparses
)
t_loss = nn.CrossEntropyLoss(ignore_index=-1)
n_loss = nn.CrossEntropyLoss(ignore_index=-1)

class DeepHitDiscreteLoss(nn.Module):
    """Loss de survie discrète façon DeepHit : vraisemblance négative sur les bins
    (événement vs censure) plus un terme de ranking par paires de patients."""

    def __init__(self, alpha: float = 0.2, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def time_to_bin(times: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
        T = bin_edges.numel() - 1
        idx = torch.bucketize(times, bin_edges[1:-1], right=False)
        return idx.clamp(0, T - 1).long()

    def forward(self, logits: torch.Tensor, times: torch.Tensor,
                events: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
        device = logits.device
        B, T = logits.shape
        pmf = torch.softmax(logits, dim=-1)
        cif = torch.cumsum(pmf, dim=-1)
        k = self.time_to_bin(times, bin_edges)
        events_b = events.bool()
        eps = 1e-8
        p_event = pmf.gather(1, k.unsqueeze(1)).squeeze(1)
        s_censor = 1.0 - cif.gather(1, k.unsqueeze(1)).squeeze(1)
        nll_event = -torch.log(p_event[events_b].clamp_min(eps)).sum() \
                    if events_b.any() else torch.tensor(0.0, device=device)
        nll_censor = -torch.log(s_censor[~events_b].clamp_min(eps)).sum() \
                     if (~events_b).any() else torch.tensor(0.0, device=device)
        l_nll = (nll_event + nll_censor) / max(B, 1)
        if events_b.any():
            cif_at_ki = cif.gather(1, k.unsqueeze(1))
            ki_expand = k.view(1, B).expand(B, B)
            cif_j_at_ki = cif.unsqueeze(0).expand(B, B, T).gather(2, ki_expand.unsqueeze(-1)).squeeze(-1)
            cif_i_at_ki = cif_at_ki.expand(B, B)
            time_mat = times.view(B, 1) < times.view(1, B)
            valid = events_b.view(B, 1) & time_mat
            diff = cif_i_at_ki - cif_j_at_ki
            rank_loss = torch.exp(-diff / self.sigma)
            rank_loss = (rank_loss * valid.float()).sum() / (valid.float().sum() + eps)
        else:
            rank_loss = torch.tensor(0.0, device=device)
        return l_nll + self.alpha * rank_loss

class UncertaintyWeightedLoss(nn.Module):
    """Combine plusieurs pertes de tâches en les pondérant par une incertitude apprise
    (homoscédastique), via des log-variances entraînables par tâche."""

    def __init__(self, n_tasks: int = 4, log_sigma_min: float = -3.0,
                 log_sigma_max: float = 3.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, *task_losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(task_losses) == self.log_sigma.numel(), \
            f"Expected {self.log_sigma.numel()} losses, got {len(task_losses)}"
        log_sigma = self.log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
        sigma_sq = torch.exp(2 * log_sigma)
        weights = 1.0 / (2.0 * sigma_sq)
        total = sum(w * l + s for w, l, s in
                    zip(weights, task_losses, log_sigma))
        return total, weights.detach()

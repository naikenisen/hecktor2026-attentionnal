import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config
from src.networks import ClinicalModel
from src.clinical_data import ClinicalDataModule
from src.image_data import BottleneckExtractor
from utils.losses import DeepHitDiscreteLoss, UncertaintyWeightedLoss
from utils.metrics import balanced_accuracy, discrete_risk_from_surv_logits, c_index

class ClinicalTrainer:

    def __init__(self, config, device, data: ClinicalDataModule):
        self.config = config
        self.device = device
        self.data = data
        self.model = ClinicalModel(config).to(device)
        self.weighting = UncertaintyWeightedLoss(n_tasks=3).to(device)
        self.deephit = DeepHitDiscreteLoss(alpha=0.2).to(device)
        self.bin_edges = data.survival_bin_edges().to(device)
        self.ce_t = nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=data.train.class_weights("t_label", config.num_t_classes, device),
        )
        self.ce_n = nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=data.train.class_weights("n_label", config.num_n_classes, device),
        )
        self.trainable_parameters = list(self.model.parameters()) + list(self.weighting.parameters())
        self.optimizer = optim.AdamW(
            self.trainable_parameters, lr=config.clinical_lr, weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, total_iters=config.clinical_epochs, power=config.poly_lr_power,
        )
        print(f"clinical heads have {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def _batch_loss(self, batch: dict) -> torch.Tensor:
        out = self.model(batch["bottleneck"], batch["clinical"])
        loss_t = self.ce_t(out["t_logits"], batch["t_label"])
        loss_n = self.ce_n(out["n_logits"], batch["n_label"])
        observed = batch["time"] > 0
        if observed.any():
            loss_survival = self.deephit(
                out["surv_logits"][observed], batch["time"][observed],
                batch["event"][observed], self.bin_edges,
            )
        else:
            loss_survival = torch.tensor(0.0, device=self.device)
        loss, _ = self.weighting(loss_t, loss_n, loss_survival)
        return loss

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in self.data.train.batches(self.config.clinical_batch_size, self.device, shuffle=True):
            loss = self._batch_loss(batch)
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.config.grad_clip_norm)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        t_true, t_pred, n_true, n_pred = [], [], [], []
        risks, times, events = [], [], []
        for batch in self.data.val.batches(self.config.clinical_batch_size, self.device, shuffle=False):
            out = self.model(batch["bottleneck"], batch["clinical"])
            t_true.extend(batch["t_label"].cpu().tolist())
            t_pred.extend(out["t_logits"].argmax(1).cpu().tolist())
            n_true.extend(batch["n_label"].cpu().tolist())
            n_pred.extend(out["n_logits"].argmax(1).cpu().tolist())
            risks.extend(discrete_risk_from_surv_logits(out["surv_logits"]).cpu().tolist())
            times.extend(batch["time"].cpu().tolist())
            events.extend(batch["event"].cpu().tolist())
        risks, times, events = np.array(risks), np.array(times), np.array(events)
        observed = times > 0
        return {
            "bal_t": balanced_accuracy(np.array(t_true), np.array(t_pred)),
            "bal_n": balanced_accuracy(np.array(n_true), np.array(n_pred)),
            "c_index": c_index(risks[observed], times[observed], events[observed]) if observed.any() else 0.5,
        }

    def fit(self):
        best_score = 0.0
        for epoch in range(self.config.clinical_epochs):
            train_loss = self.train_epoch()
            print(f"epoch {epoch+1}/{self.config.clinical_epochs} loss {train_loss:.4f}")
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.config.clinical_epochs:
                metrics = self.evaluate()
                print(f"validation balacc T {metrics['bal_t']:.4f} "
                      f"balacc N {metrics['bal_n']:.4f} c-index {metrics['c_index']:.4f}")
                score = (metrics["bal_t"] + metrics["bal_n"] + metrics["c_index"]) / 3
                if score > best_score:
                    best_score = score
                    torch.save(self.model.state_dict(), self.config.best_clinical_path)
                    print(f"saved new best clinical model (score {score:.4f}) to {self.config.best_clinical_path}")
            self.scheduler.step()
        print("clinical training complete")

def main():
    device = torch.device("cuda")
    print(f"using device {device} - {torch.cuda.get_device_name(device)}")
    for directory in (config.experiment_dir, config.checkpoint_dir):
        os.makedirs(directory, exist_ok=True)
    BottleneckExtractor(config, device).run()
    data = ClinicalDataModule(config).setup()
    config.n_clinical_features = data.clinical_dim
    print(f"clinical feature dimension is {data.clinical_dim}")
    ClinicalTrainer(config, device, data).fit()

if __name__ == "__main__":
    main()

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from config_bwe import BWEConfig
from losses_bwe import CombinedLoss, compute_snr, compute_lsd

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BWEConfig.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


class BWETrainer:
    def __init__(self, model, device=None):
        BWEConfig.init_env()
        BWEConfig.set_seed()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.criterion = CombinedLoss(
            l1_weight=BWEConfig.L1_WEIGHT,
            stft_weight=BWEConfig.STFT_WEIGHT,
            fft_sizes=BWEConfig.STFT_SCALES,
            hop_ratios=BWEConfig.STFT_HOP_RATIOS,
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=BWEConfig.LR, weight_decay=BWEConfig.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10,
        )

        os.makedirs(BWEConfig.CHECKPOINT_DIR, exist_ok=True)
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.no_improve = 0

    def _train_epoch(self, loader):
        self.model.train()
        total_loss, count = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch_x)
            loss, _ = self.criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), BWEConfig.GRAD_CLIP)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        return total_loss / count

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total_loss, count = 0.0, 0
        snrs, lsds = [], []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred = self.model(batch_x)
            loss, _ = self.criterion(pred, batch_y)
            total_loss += loss.item()
            count += 1
            for i in range(batch_x.size(0)):
                snrs.append(compute_snr(pred[i], batch_y[i]))
                lsds.append(compute_lsd(pred[i], batch_y[i]))
        return (total_loss / count,
                np.mean(snrs), np.mean(lsds))

    def fit(self, train_loader, val_loader, epochs=None):
        if epochs is None:
            epochs = BWEConfig.EPOCHS

        logger.info(f"Training AudioUNet on {self.device}, max {epochs} epochs")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, snr, lsd = self._validate(val_loader)
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.no_improve = 0
                self._save("best_model_bwe.pth", epoch)
                marker = "*"
            else:
                self.no_improve += 1
                marker = " "

            logger.info(
                f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
                f"| SNR: {snr:.2f}dB | LSD: {lsd:.4f} | LR: {lr:.2e} {marker}"
            )

            if self.no_improve >= BWEConfig.EARLY_STOP_PATIENCE:
                logger.info(
                    f"Early stop at epoch {epoch} "
                    f"(best: epoch {self.best_epoch}, val_loss={self.best_val_loss:.4f})"
                )
                break

        self._load("best_model_bwe.pth")
        return self.best_val_loss

    def evaluate(self, loader):
        self._load("best_model_bwe.pth")
        self.model.eval()

        total_loss, count = 0.0, 0
        snrs, lsds = [], []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred = self.model(batch_x)
            loss, _ = self.criterion(pred, batch_y)
            total_loss += loss.item()
            count += 1
            for i in range(batch_x.size(0)):
                snrs.append(compute_snr(pred[i], batch_y[i]))
                lsds.append(compute_lsd(pred[i], batch_y[i]))

        return {
            "loss": total_loss / count,
            "snr_db": np.mean(snrs),
            "lsd": np.mean(lsds),
        }, snrs, lsds

    def _save(self, filename, epoch):
        path = os.path.join(BWEConfig.CHECKPOINT_DIR, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epoch": epoch,
        }, path)

    def _load(self, filename):
        path = os.path.join(BWEConfig.CHECKPOINT_DIR, filename)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

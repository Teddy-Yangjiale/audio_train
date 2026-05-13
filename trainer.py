import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from config import Config

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def build_loaders(X_train, y_train, X_val, y_val, X_test=None, y_test=None,
                  batch_size=None, augment=False):
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if augment:
        train_ds = AugmentedTensorDataset(X_train, y_train)
    else:
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=batch_size, shuffle=False,
    )
    if X_test is not None and y_test is not None:
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=batch_size, shuffle=False,
        )
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


class AugmentedTensorDataset(Dataset):
    def __init__(self, X, y, augment_prob=0.5):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.prob = augment_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if np.random.random() < self.prob:
            from augmentation import spec_augment
            x_np = x.numpy().T
            x_np = spec_augment(x_np)
            x = torch.FloatTensor(x_np.T)

        return x, y


def _compute_class_weights(y_train):
    """Compute inverse-frequency class weights."""
    counts = np.bincount(y_train)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights)


class Trainer:
    def __init__(self, model, device=None, class_weights=None,
                 label_smoothing=None, scheduler_type=None):
        Config.init_env()
        Config.set_seed()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # --- Loss ---
        ls = label_smoothing if label_smoothing is not None else Config.LABEL_SMOOTHING
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=ls,
        )

        # --- Optimizer ---
        self.optimizer = optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY,
        )

        # --- Scheduler ---
        sched = scheduler_type or Config.SCHEDULER
        if sched == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=Config.COSINE_T_MAX,
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=Config.LR_FACTOR,
                patience=Config.LR_PATIENCE,
            )
        self.scheduler_type = sched

        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.start_epoch = 0

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            outputs = self.model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        return correct / total

    def fit(self, train_loader, val_loader, epochs=None, resume=None):
        if resume:
            self._resume(resume)
        if epochs is None:
            epochs = Config.EPOCHS

        logger.info(f"Training on {self.device} from epoch {self.start_epoch + 1} for up to {epochs} epochs"
                    f" (scheduler={self.scheduler_type})")
        logger.info(f"  Label smoothing={self.criterion.label_smoothing}, "
                    f"Class weights={'enabled' if self.criterion.weight is not None else 'disabled'}")

        for epoch in range(self.start_epoch + 1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_acc = self._validate(val_loader)

            # Some schedulers need loss, some need metric
            if self.scheduler_type == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(val_acc)

            lr = self.optimizer.param_groups[0]["lr"]

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self._save("best_model.pth", epoch)
                marker = "*"
            else:
                self.epochs_no_improve += 1
                marker = " "

            logger.info(
                f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% "
                f"| LR: {lr:.2e} {marker}"
            )

            if not Config.SAVE_BEST_ONLY:
                self._save(f"epoch_{epoch:04d}.pth", epoch)

            if self.epochs_no_improve >= Config.EARLY_STOP_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch} "
                            f"(best: epoch {self.best_epoch}, {self.best_val_acc*100:.2f}%)")
                break

        self._load("best_model.pth")
        return self.best_val_acc

    @torch.no_grad()
    def evaluate(self, loader, label_names=None):
        self.model.eval()
        all_preds, all_labels = [], []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            outputs = self.model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

        from sklearn.metrics import classification_report
        report = classification_report(
            all_labels, all_preds,
            target_names=label_names if label_names is not None else None,
        )
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        return acc, report

    def _save(self, filename, epoch):
        path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "epoch": epoch,
        }, path)

    def _load(self, filename):
        path = os.path.join(Config.CHECKPOINT_DIR, filename)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

    def _resume(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        self.start_epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from {path} (epoch {self.start_epoch}, best val acc {self.best_val_acc*100:.2f}%)")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(tensor)
            return torch.argmax(outputs, dim=1).cpu().numpy()

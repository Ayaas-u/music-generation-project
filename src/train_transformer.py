import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformer import DrumTransformer


BOS_TOKEN = 64
VOCAB_SIZE = 66
BLOCK_SIZE = 128


class DrumTokenDataset(Dataset):
    def __init__(self, token_sequences):
        self.token_sequences = token_sequences

    def __len__(self):
        return len(self.token_sequences)

    def __getitem__(self, idx):
        seq = self.token_sequences[idx]
        x = np.concatenate([[BOS_TOKEN], seq[:-1]])
        y = seq.copy()

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def evaluate(model, loader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())

    mean_loss = float(np.mean(losses))
    perplexity = float(math.exp(mean_loss))
    return mean_loss, perplexity


def main():
    data_dir = Path("data")
    split_dir = data_dir / "train_test_split"

    token_sequences = np.load(data_dir / "groove_step_tokens.npy")
    train_idx = np.load(split_dir / "train_idx.npy")
    val_idx = np.load(split_dir / "val_idx.npy")
    test_idx = np.load(split_dir / "test_idx.npy")

    train_data = token_sequences[train_idx]
    val_data = token_sequences[val_idx]
    test_data = token_sequences[test_idx]

    train_dataset = DrumTokenDataset(train_data)
    val_dataset = DrumTokenDataset(val_data)
    test_dataset = DrumTokenDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DrumTransformer(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    num_epochs = 40
    best_val_loss = float("inf")
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_perplexity": [],
        "val_perplexity": [],
    }

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        train_ppl = float(math.exp(train_loss))
        val_loss, val_ppl = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_perplexity"].append(train_ppl)
        history["val_perplexity"].append(val_ppl)

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_ppl={train_ppl:.4f} | "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            }

    if best_state is not None:
        torch.save(best_state, data_dir / "transformer_model.pth")
        model.load_state_dict(best_state["model_state_dict"])

    test_loss, test_ppl = evaluate(model, test_loader, device)

    results = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_state["epoch"] if best_state is not None else None,
        "best_val_perplexity": best_state["val_perplexity"] if best_state is not None else None,
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
        "config": {
            "vocab_size": VOCAB_SIZE,
            "block_size": BLOCK_SIZE,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "dropout": 0.1,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "epochs": num_epochs,
        },
        "history": history,
    }

    with open(data_dir / "transformer_training_info.json", "w") as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Transformer Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / "transformer_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_perplexity"], label="Train Perplexity")
    plt.plot(history["val_perplexity"], label="Val Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Transformer Perplexity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / "transformer_perplexity_curve.png", dpi=200)
    plt.close()

    print("Saved model and training info.")
    print(f"Test perplexity: {test_ppl:.4f}")


if __name__ == "__main__":
    main()
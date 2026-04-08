import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMVAE(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=32, seq_len=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder prep
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        hidden = self.from_latent(z)                     # (B, H)
        repeated = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, H)
        decoded, _ = self.decoder(repeated)
        logits = self.output_layer(decoded)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


def load_data(npy_path="data/groove_sequences.npy"):
    data = np.load(npy_path)
    print(f"Loaded raw data shape: {data.shape}")

    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.shape}")

    # Convert (N, C, T) -> (N, T, C)
    if data.shape[1] == 6:
        data = np.transpose(data, (0, 2, 1))
    elif data.shape[2] == 6:
        pass
    else:
        raise ValueError(f"Cannot identify channel dimension in shape {data.shape}")

    data = data.astype(np.float32)

    # If data is not binary already, binarize lightly
    if data.max() > 1.0 or data.min() < 0.0:
        data = (data > 0).astype(np.float32)

    print(f"Standardized data shape: {data.shape}")
    return data


def vae_loss_fn(logits, target, mu, logvar, beta=0.005):
    recon_loss = nn.BCEWithLogitsLoss()(logits, target)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_vae(
    npy_path="data/groove_sequences.npy",
    model_out="data/lstm_vae.pth",
    info_out="data/vae_training_info.json",
    plot_out="data/vae_loss_curve.png",
    epochs=200,
    batch_size=8,
    lr=1e-3,
    beta=0.005
):
    data = load_data(npy_path)
    seq_len = data.shape[1]
    input_dim = data.shape[2]

    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTMVAE(
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=32,
        seq_len=seq_len
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_losses = []
    recon_losses = []
    kl_losses = []

    model.train()
    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for (x,) in loader:
            x = x.to(device)

            optimizer.zero_grad()
            logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss_fn(logits, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        avg_total = epoch_total / len(loader)
        avg_recon = epoch_recon / len(loader)
        avg_kl = epoch_kl / len(loader)

        total_losses.append(avg_total)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Total: {avg_total:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}"
            )

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"Saved VAE model to: {model_out}")

    training_info = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "beta": beta,
        "final_total_loss": total_losses[-1],
        "final_recon_loss": recon_losses[-1],
        "final_kl_loss": kl_losses[-1],
    }

    with open(info_out, "w") as f:
        json.dump(training_info, f, indent=2)
    print(f"Saved training info to: {info_out}")

    plt.figure(figsize=(8, 5))
    plt.plot(total_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_out)
    plt.close()
    print(f"Saved loss plot to: {plot_out}")


if __name__ == "__main__":
    train_vae()
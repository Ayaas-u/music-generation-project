import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

# ======================
# Model Definition
# ======================

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=32):
        super(LSTMAutoencoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]
        z = self.fc_latent(h)
        return z

    def from_latent(self, z):
        return self.fc_decode(z)

    def decode(self, z):
        # Expand latent across time steps
        z_expanded = z.unsqueeze(1).repeat(1, 128, 1)
        out, _ = self.decoder(z_expanded)
        return out

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(self.from_latent(z))
        out = torch.sigmoid(self.output_layer(decoded))
        return out


# ======================
# Training Function
# ======================

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load dataset
    data = np.load("data/groove_sequences.npy")
    print(f"Loaded data shape: {data.shape} (Samples, Channels, Time)")

    # Convert to (samples, time, channels)
    data = np.transpose(data, (0, 2, 1))
    data = torch.tensor(data, dtype=torch.float32).to(device)

    print(f"Converted shape: {data.shape} (Samples, Timesteps, Drums)")

    # Model
    model = LSTMAutoencoder(input_dim=6, hidden_dim=64, latent_dim=32).to(device)

    # Loss & optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    batch_size = 64

    losses = []

    print("\nStarting training...")

    for epoch in range(epochs):
        perm = torch.randperm(data.size(0))
        epoch_loss = 0

        for i in range(0, data.size(0), batch_size):
            indices = perm[i:i+batch_size]
            batch = data[indices]

            optimizer.zero_grad()

            outputs = model(batch)
            loss = criterion(outputs, batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= (data.size(0) // batch_size)
        losses.append(epoch_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")

    # Save model
    os.makedirs("data", exist_ok=True)
    torch.save(model.state_dict(), "data/lstm_autoencoder.pth")
    print("\n✓ Model saved to ./data/lstm_autoencoder.pth")

    # Save loss curve
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("data/loss_curve.png")
    print("✓ Loss curve saved to data/loss_curve.png")


# ======================
# Run
# ======================

if __name__ == "__main__":
    train_model()
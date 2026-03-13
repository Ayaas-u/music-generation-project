import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

# Check for GPU (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# --- 1. DATA LOADING ---
def load_and_prepare_data(filepath):
    """Loads the sequences and prepares them for the LSTM (Samples, Timesteps, Features)"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ {filepath} not found! Run midi_to_pianoroll.py first.")
    
    # Load sequences: (Samples, Drums, Timesteps) -> (10, 6, 128)
    data = np.load(filepath)
    
    # LSTM expects (Batch, Seq_Len, Input_Dim) -> (10, 128, 6)
    data = np.transpose(data, (0, 2, 1))
    
    # Convert to PyTorch Tensor and move to GPU/CPU
    return torch.tensor(data, dtype=torch.float32).to(device)

# --- 2. MODEL DEFINITION ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=32):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder: Compresses the 6-drum input into a hidden state
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Expands the latent vector back into a sequence
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Encoding
        _, (hidden, _) = self.encoder(x)
        z = self.to_latent(hidden[-1])  # z is our 'musical essence'

        # Decoding
        # We repeat 'z' for every timestep so the decoder knows what to build
        z_expanded = self.from_latent(z).unsqueeze(1).repeat(1, seq_len, 1)
        reconstruction, _ = self.decoder(z_expanded)
        
        # Map to 0-1 range (probabilities of a drum hit)
        x_hat = torch.sigmoid(self.output_layer(reconstruction))
        return x_hat

# --- 3. TRAINING LOOP ---
def train_model():
    # Parameters
    INPUT_PATH = 'data/groove_sequences.npy'
    MODEL_SAVE_PATH = './data/lstm_autoencoder.pth'
    EPOCHS = 200
    LEARNING_RATE = 0.001

    # Load Data
    X = load_and_prepare_data(INPUT_PATH)
    print(f"Loaded data shape: {X.shape} (Samples, Timesteps, Drums)")

    # Initialize Model, Loss, and Optimizer
    model = LSTMAutoencoder(input_dim=6, hidden_dim=64, latent_dim=32).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X)
        loss = criterion(output, X)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {loss.item():.6f}")

    # Save the trained weights
    os.makedirs('data', exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")

    # Optional: Plot the reconstruction loss curve (Requirement from your project list!)
    plt.plot(history)
    plt.title('Reconstruction Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.savefig('data/loss_curve.png')
    print("✓ Loss curve saved to data/loss_curve.png")

if __name__ == "__main__":
    train_model()
import numpy as np

data = np.load("data/vae_generated_samples.npy")

print("Shape:", data.shape)
print("Min:", data.min())
print("Max:", data.max())
print("Mean:", data.mean())
print("Std:", data.std())

print("\n--- Threshold Sweep (0.40 → 0.60) ---")

thresholds = np.linspace(0.4, 0.6, 30)
target_density = 0.1756  # real dataset density from latest evaluation

results = []

for t in thresholds:
    binary = (data > t).astype(int)
    density = binary.sum() / binary.size
    results.append((t, density))
    print(f"Threshold {t:.3f} -> density {density:.4f}")

best_t = min(results, key=lambda x: abs(x[1] - target_density))

print("\n--- Best Threshold Match ---")
print(f"Threshold {best_t[0]:.3f} gives density {best_t[1]:.4f}")
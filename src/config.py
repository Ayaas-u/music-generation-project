from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Core folders
DATA_DIR = ROOT_DIR / "data"
RAW_MIDI_DIR = DATA_DIR / "raw_midi" / "groove"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "train_test_split"
SURVEY_RESULTS_DIR = ROOT_DIR / "survey_results"

OUTPUTS_DIR = ROOT_DIR / "outputs"
GENERATED_MIDIS_DIR = OUTPUTS_DIR / "generated_midis"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

REPORT_DIR = ROOT_DIR / "report"
ARCHITECTURE_DIR = ROOT_DIR / "architecture_diagrams"

# Data files
GROOVE_SEQUENCES_PATH = PROCESSED_DIR / "groove_sequences.npy"
GROOVE_STEP_TOKENS_PATH = PROCESSED_DIR / "groove_step_tokens.npy"
GROOVE_STEP_TOKENS_INFO_PATH = PROCESSED_DIR / "groove_step_tokens_info.json"

TRAIN_IDX_PATH = SPLIT_DIR / "train_idx.npy"
VAL_IDX_PATH = SPLIT_DIR / "val_idx.npy"
TEST_IDX_PATH = SPLIT_DIR / "test_idx.npy"
SPLIT_INFO_PATH = SPLIT_DIR / "split_info.json"

# Model checkpoints
AE_MODEL_PATH = CHECKPOINTS_DIR / "lstm_autoencoder.pth"
VAE_MODEL_PATH = CHECKPOINTS_DIR / "lstm_vae.pth"
TRANSFORMER_MODEL_PATH = CHECKPOINTS_DIR / "transformer_model.pth"
RLHF_MODEL_PATH = CHECKPOINTS_DIR / "rlhf_model.pth"
REWARD_MODEL_PATH = CHECKPOINTS_DIR / "reward_model.npz"

# Training info
AE_TRAINING_INFO_PATH = CHECKPOINTS_DIR / "training_info.json"
VAE_TRAINING_INFO_PATH = CHECKPOINTS_DIR / "vae_training_info.json"
TRANSFORMER_TRAINING_INFO_PATH = CHECKPOINTS_DIR / "transformer_training_info.json"
RLHF_TRAINING_INFO_PATH = CHECKPOINTS_DIR / "rlhf_training_info.json"

# Plot files
AE_LOSS_PLOT_PATH = PLOTS_DIR / "loss_curve.png"
VAE_LOSS_PLOT_PATH = PLOTS_DIR / "vae_loss_curve.png"
TRANSFORMER_LOSS_PLOT_PATH = PLOTS_DIR / "transformer_loss_curve.png"
TRANSFORMER_PPL_PLOT_PATH = PLOTS_DIR / "transformer_perplexity_curve.png"

# Drum setup
NUM_DRUM_CHANNELS = 6
DRUM_LABELS = ["Kick", "Snare", "HiHat", "Tom", "Crash", "Ride"]

DRUM_NOTES = {
    0: 36,  # Kick
    1: 38,  # Snare
    2: 42,  # HiHat
    3: 45,  # Tom
    4: 49,  # Crash
    5: 51,  # Ride
}

# Sequence / token settings
SEQUENCE_LENGTH = 128
BLOCK_SIZE = 128
VOCAB_SIZE = 66
BOS_TOKEN = 64

# Default generation settings
DEFAULT_TEMPO = 120
DEFAULT_VELOCITY = 115
DEFAULT_STEP_DURATION = 0.125

# Transformer defaults
TRANSFORMER_D_MODEL = 128
TRANSFORMER_N_HEADS = 4
TRANSFORMER_N_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1

# Training defaults
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 40
SEED = 42
Before the readme part I would like to express my  gratitude to our faculty, MMM, for assigning this project. Although it was extensive, the topic was  engaging and provided valuable hands-on experience. It gave me the opportunity to work with multiple models and understand how to implement and tune them effectively. This experience has been very beneficial for my learning and practical understanding.
# 🎵 Music Generation with LSTM Autoencoders
# Music Generation Unsupervised

A complete neural music-generation course project built on the **Groove MIDI Dataset**, focusing on **symbolic drum and groove generation** with modern sequence models.

This project explores how different generative models learn rhythmic structure from MIDI drum data and compares their ability to produce musically meaningful drum patterns across four stages:

- **Task 1:** LSTM Autoencoder
- **Task 2:** LSTM Variational Autoencoder (VAE)
- **Task 3:** Transformer-based drum generation
- **Task 4:** RLHF-style preference tuning

The work uses a **6-channel drum representation**:

- Kick
- Snare
- Hi-Hat
- Tom
- Crash
- Ride

The assignment requires a public MIDI dataset, preprocessing and train-test split, baselines, evaluation metrics, generated samples, and a final report, all of which are covered in this project.

## Project Motivation

Rather than treating music generation as a generic sequence problem, this project focuses on **groove and rhythm**, where timing, repetition, variation, and drum interaction are the key musical ingredients.

Using the Groove MIDI dataset, the project investigates how progressively stronger models move from simple reconstruction to more expressive generation:

- the **LSTM Autoencoder** learns compressed rhythmic structure
- the **VAE** adds latent sampling for more variation
- the **Transformer** improves long-range pattern modeling through token-based autoregressive generation
- the **RLHF-style tuning stage** experiments with human preference-driven improvement

## Dataset

- **Dataset:** Groove MIDI Dataset
- **Type:** Public MIDI dataset
- **Representation:** symbolic 6-channel drum sequences

The project uses Groove as its public MIDI source and adapts the assignment into a groove-focused symbolic drum-generation setup. 

## Pipeline Overview

The overall pipeline is:

1. **Parse Groove MIDI files**
2. **Convert to 6-channel symbolic drum representation**
3. **Create train / validation / test splits**
4. **Train generative models**
5. **Generate new drum sequences**
6. **Export generated MIDI**
7. **Evaluate outputs using symbolic metrics**
8. **Apply RLHF-style tuning for Task 4**

## Tasks Completed

### Task 1 — LSTM Autoencoder
A sequence autoencoder trained on drum piano-roll windows to reconstruct groove patterns.

### Task 2 — LSTM VAE
An extension of the autoencoder with:
- latent mean and log-variance
- reparameterization trick
- KL divergence regularization

This improves variation compared to the basic autoencoder. 

### Task 3 — Transformer
A decoder-only Transformer trained on tokenized drum-step patterns.

Each timestep is encoded as a discrete token representing the 6 drum channels, and generation is performed autoregressively with temperature and top-k sampling. This became the strongest model in the project because token-based generation gave more lively and varied outputs than thresholded AE/VAE reconstruction. 

### Task 4 — RLHF-style Preference Tuning
A lightweight reinforcement-learning-from-human-feedback style pipeline built around:
- human survey data
- reward scoring / reward modeling
- RL fine-tuning
- tuned sample generation
- before-vs-after comparison

This stage follows the assignment’s requirement for preference-based tuning in a compact educational setup.

## Baselines

To provide meaningful comparison, the project also includes:
- **Random drum generation baseline**
- **Markov chain baseline**

These help contextualize the performance of the neural models. 
## Evaluation

The main evaluation metrics used throughout the project are:

- **Density**
- **Diversity**
- **Repetition Ratio**

These metrics are used to compare:
- real dataset sequences
- random baseline
- Markov baseline
- LSTM Autoencoder
- LSTM VAE
- Transformer
- RLHF-tuned Transformer 

For this drum-focused version:
- `metrics.py` contains the core symbolic metrics
- `rhythm_score.py` provides drum-oriented rhythm evaluation
- `pitch_histogram.py` is adapted as **drum-hit distribution** across the 6 drum channels rather than melodic pitch classes

## Repository Structure

```text
music-generation-unsupervised/
README.md
requirements.txt
data/
  raw_midi/
  processed/
  train_test_split/
notebooks/
  preprocessing.ipynb
  baseline_markov.ipynb
src/
  config.py
  preprocessing/
    midi_parser.py
    tokenizer.py
    piano_roll.py
  models/
    autoencoder.py
    vae.py
    transformer.py
    diffusion.py
  training/
    train_ae.py
    train_vae.py
    train_transformer.py
    train_reward_model.py
    train_rlhf.py
    aggregate_human_rewards.py
    prepare_google_form_ratings.py
  evaluation/
    metrics.py
    pitch_histogram.py
    rhythm_score.py
    evaluate_models.py
    evaluate_transformer.py
    compare_before_after_rlhf.py
  generation/
    sample_latent.py
    generate_music.py
    generate_vae_midi.py
    generate_rlhf_midi.py
    midi_export.py
outputs/
  generated_midis/
  plots/
  checkpoints/
survey_results/
report/
  final_report.tex
  references.bib
  architecture_diagrams/

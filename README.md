# Raspberry Pi FNN: Signal Windowing and Autoencoder Pipeline

This repository provides a complete workflow for training a lightweight autoencoder, exporting the encoder to a Raspberry Pi, generating or collecting signal data on the Pi, producing latent representations, and decoding them on a desktop machine for reconstruction and evaluation.

## Overview

The system performs the following steps:

1. Segment raw signals into fixed-length windows such as 512 samples.
2. Apply per-window z-normalization.
3. Train an autoencoder in PyTorch.
4. Export the encoder to the Raspberry Pi for real-time inference.
5. Log latent vectors on the Pi.
6. Transfer latent logs to a desktop machine.
7. Use the decoder to reconstruct the signals and compute reconstruction loss.

## Installation

Install dependencies on a standard machine:
``pip install -r requirements.txt``

Install CPU PyTorch on the Raspberry Pi:
``pip install torch –index-url https://download.pytorch.org/whl/cpu``

## Training
``python scripts/train.py –epochs 50 –window-size 512 –save-path encoder.pt``

This produces both encoder and decoder model files.

## Raspberry Pi Inference

Run the encoder on the Pi to generate latent vectors and store them locally:
``python scripts/pi_infer.py –model encoder.pt –log-dir latents/``

To transfer logs back to your desktop:
``scp pi@:~/latents/*.npy ./latents/``

## Desktop Decoding

Use the decoder to reconstruct signals from latent vectors:
``python scripts/decode.py –model decoder.pt –latents latents/``

This produces reconstructed signals, optional plots, and reconstruction loss values.

## Synthetic Data

If no real sensor is available, generate synthetic signals that match the training format:
``python scripts/generate.py –num 10000 –out data/synth.npy``
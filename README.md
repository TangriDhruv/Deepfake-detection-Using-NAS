# Audio Deepfake Detection with Neural Architecture Search

This repository contains a modular implementation of a deepfake detection system for audio, specifically designed to detect spoofing attacks in automatic speaker verification (ASV) systems. The approach uses Neural Architecture Search (NAS) to automatically discover optimal neural network architectures that operate directly on raw audio waveforms.

## Project Overview

The system implements a hybrid approach combining Proximal Policy Optimization (PPO) and Differentiable Architecture Search (DARTS) to efficiently discover neural network architectures that are effective for audio deepfake detection. The implementation is evaluated on the ASVspoof 2019 Logical Access dataset.

### Key Features

- **End-to-end learning**: Operates directly on raw audio waveforms without requiring hand-crafted features
- **Hybrid NAS approach**: Combines PPO and DARTS for better exploration/exploitation balance
- **Modular design**: Clean separation between data handling, model architecture, search algorithms, and evaluation
- **Visualization tools**: Comprehensive architecture visualization for better interpretability

## Dataset Preparation
The code is designed to work with the ASVspoof 2019 Logical Access dataset. Follow these steps to prepare the dataset:

Download the ASVspoof 2019 LA dataset from the official website
Extract the dataset to a suitable location
Update the dataset paths in config.py to point to your dataset location

## Running Neural Architecture Search
python main.py

## Using Weights & Biases for Experiment Tracking
The code integrates with Weights & Biases (wandb) for experiment tracking. To use wandb:

Sign up for a free account at wandb.ai
Get your API key from the wandb website
Add your API key to the config.py file

## Acknowledgements

The ASVspoof 2019 challenge organizers for providing the dataset
The authors of the DARTS and PC-DARTS papers for the architecture search methods
The authors of RawNet for inspiration on raw waveform processing

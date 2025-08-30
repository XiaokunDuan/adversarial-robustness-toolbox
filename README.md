# Adversarial Robustness Toolbox

A collection of hands-on experiments demonstrating key techniques for improving AI model robustness, implemented in PyTorch.

This repository is a practical guide to understanding and implementing three fundamental directions in adversarial robustness.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Expected Results](#expected-results)

## Core Concepts

This toolbox contains implementations for:

1.  **Direction 1: Adversarial Training**: Defending against small, malicious perturbations by training the model on adversarial examples (using FGSM).
2.  **Direction 2: GAN for Data Augmentation**: Using a Generative Adversarial Network to create a diverse training set to improve generalization robustness against natural variations like rotation.
3.  **Direction 3: Domain-Adversarial Training (DANN)**: Bridging the "domain gap" by learning domain-invariant features, adapting a model from MNIST to a colored MNIST version without using target labels.

## Project Structure
To be modified

## Setup and Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/XiaokunDuan/adversarial-robustness-toolbox.git
    cd adversarial-robustness-toolbox
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

You can run each experiment independently.

-   **Adversarial Training Demo**:
    ```bash
    python dir1_adversarial_training/run_experiment.py
    ```
-   **GAN Augmentation Demo**:
    ```bash
    python dir2_gan_augmentation/run_experiment.py
    ```
-   **Domain Adaptation Demo**:
    ```bash
    python dir3_domain_adaptation/run_experiment.py
    ```

## Expected Results

After running the experiments, you will find visualizations and model weights in the `results/` directory of each direction.

**Example: DANN Feature Space Alignment**


still on the pipe
---
**License**: MIT

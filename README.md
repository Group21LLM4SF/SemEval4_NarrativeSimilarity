# SemEval4 - Narrative Similarity

A deep learning approach to **narrative similarity** using aspect-aware triplet learning. This project implements a model that learns to compare narratives based on semantic aspects such as theme, course of action, and outcomes.

## Overview

This repository contains the implementation for [SemEval-2026 Task 4: Narrative Similarity](https://narrative-similarity-task.github.io/) The approach leverages:

- **Triplet Learning**: Learning embeddings where similar narratives are closer together
- **Aspect-Aware Attention**: Multi-head cross-attention on narrative aspects (theme, action, outcomes)
- **LLM-Based Data Augmentation**: Using Llama 3.3 70B to generate augmented training triplets
- **LoRA Fine-Tuning** (optional): Parameter-efficient fine-tuning of the encoder

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── encoder.py        # AspectSupervisedEncoder model
│   │   ├── attention.py      # Cross-attention modules for aspects
│   │   └── loss.py           # Triplet, Aspect MSE, and Cross losses
│   ├── datasets/
│   │   └── dataset.py        # Dataset classes for triplet data
│   ├── llm/
│   │   ├── augmentor.py      # LLM-based triplet augmentation
│   │   ├── aspects_extractor.py  # Extract aspects from narratives
│   │   └── prompts/          # Prompt templates
│   └── trainer.py            # Training loop and evaluation
├── notebook/
│   ├── training.ipynb        # Model training notebook
│   └── data_preprocessing.ipynb  # Data augmentation pipeline
└── data/
    ├── raw/                  # Original track data
    └── processed/            # Processed triplets with aspects
```

## Model Architecture

**AspectSupervisedEncoder** consists of:

1. **Base Encoder**: Pre-trained transformer (DistilBERT/BERT/MPNet) with optional LoRA
2. **Projection Head**: Maps encoder outputs to a unified embedding space
3. **Aspect Heads**: Separate prediction heads for theme, action, and outcome
4. **Cross-Attention Module**: Refines aspect representations using multi-head attention

## Loss Functions

The combined loss function:

$$L_{total} = L_{triplet} + \gamma \cdot L_{aspect} + \beta \cdot L_{cross}$$

- **Triplet Loss**: Ensures anchor-positive similarity > anchor-negative similarity
- **Aspect MSE Loss**: Supervises aspect head predictions
- **Aspect Cross Loss**: Triplet loss applied to aspect embeddings

## Installation

```bash
pip install -r requirements.txt
```

**Key dependencies**: PyTorch, Transformers, PEFT, Together AI, Datasets

## Usage

### Data Preprocessing

```python
from src.llm.augmentor import AugmentTripletGenerator

generator = AugmentTripletGenerator(api_key="your-key")
triplets = generator.run_batch(dev_data, n_triplets_per_sample=2)
```

### Training

```python
from src.models.encoder import AspectSupervisedEncoder
from src.trainer import Trainer

model = AspectSupervisedEncoder(
    model_name="distilbert-base-uncased",
    projection_dim=256,
    aspect_dim=128,
    num_heads=4
)

trainer = Trainer(model, device, lr=1e-4)
trainer.fit(train_loader, eval_loader, epochs=10)
```

## Data Tracks

- **Track A**: Narrative similarity with labeled pairs
- **Track B**: Additional evaluation data



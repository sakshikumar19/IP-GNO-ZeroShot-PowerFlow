# IP-GNO v4: Physics-Informed Graph Neural Operators for Power Flow Prediction

This project implements IP-GNO (Physics-Informed Graph Neural Operator), a machine learning model for predicting voltage magnitudes and angles in electrical distribution grids. It combines graph neural networks with physics-based corrections to achieve accurate power flow solutions.

## Key Features

- **Graph Neural Operators**: Learns the mapping from load conditions to voltage solutions on graph-structured power grids
- **Physics-Informed**: Incorporates LinDistFlow equations for improved accuracy and physical consistency
- **Cross-Grid Transfer**: Models trained on one grid can predict on unseen grids without fine-tuning
- **PETE Features**: Physics-Embedded Tree Encoding provides scale-invariant positional features

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Training
Train models on the IEEE 33-bus system:
```bash
python train.py --grid 33 --model all
```

### Evaluation
Evaluate trained models:
```bash
python evaluate.py
```

### Cross-Grid Evaluation
Test models trained on 33-bus on the 69-bus grid:
```bash
python evaluate_crossgrid.py --train_grid 33 --test_grid 69 --train_outdir outputs_33
```

## Project Structure

- `train.py`: Main training script for all models
- `evaluate.py`: Evaluation on the same grid used for training
- `evaluate_crossgrid.py`: Zero-shot evaluation on unseen grids
- `data/`: Dataset generation and grid simulators
  - `dataset.py`: Core dataset utilities and PETE feature computation
  - `ieee33.py`: IEEE 33-bus system simulator
  - `ieee69.py`: IEEE 69-bus system simulator
- `models/`: Model implementations
  - `gno.py`: Vanilla GNO baseline
  - `ip_gno.py`: Physics-informed GNO with admittance kernel
  - `distflow.py`: LinDistFlow correction layer
  - `kernel_layer.py`: GNO kernel implementations
  - `baselines.py`: Traditional GNN baselines (GAT, GINE, SAGE)

## Models

- **Vanilla GNO**: Standard graph neural operator using impedance features
- **IP-GNO**: Physics-informed version with admittance-based kernels and DistFlow correction
- **Baselines**: GAT, GINE, and GraphSAGE variants for comparison

## Key Innovations

1. **PETE (Physics-Embedded Tree Encoding)**: Scale-invariant positional features based on electrical path impedances
2. **Admittance Kernel**: Uses electrical admittance instead of impedance for better physical grounding
3. **Per-Graph Normalization**: Normalizes edge features within each graph for cross-grid transfer
4. **DistFlow Correction**: Post-processes GNO predictions using linearized power flow equations

## Results

The models achieve high accuracy on voltage prediction tasks, with IP-GNO showing superior performance and better physical consistency compared to traditional GNNs.
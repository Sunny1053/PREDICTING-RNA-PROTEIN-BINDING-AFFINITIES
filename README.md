# CHE 629 Project: Deep Learning for Predicting RNA–Protein Binding (DLPRB)

## Overview

This project replicates and extends the DLPRB framework for predicting RNA–protein binding affinities using deep learning. The work progressed through three iterative stages — from a simple proof-of-concept on synthetic data, to a full-scale reproduction using real experimental data — evaluating both CNN and RNN architectures and studying the contribution of RNA secondary structure to prediction quality.

---

## Project Evolution

### Approach 1 — Synthetic Data, Sequence-Only Baseline

**Notebook:** `che629_project_firstApproach.ipynb`

**Goal:** Establish a working deep learning pipeline for RNA binding prediction on controlled synthetic data.

**Dataset:**
- 40,000 synthetic RNA sequences of length 40, generated randomly from {A, C, G, U}
- Binding scores computed from a motif-based formula:
  ```
  score = (count of "AUGC" motif) × 1.5 + GC_content + N(0, 0.2)
  ```
- Scores normalized (z-score) and clipped to [−3, 3]

**Encoding:**
- 4-dimensional one-hot encoding per nucleotide: `A → [1,0,0,0]`, `C → [0,1,0,0]`, etc.
- Input shape: `(N, 40, 4)`

**Models (PyTorch):**

| Model | Architecture |
|-------|-------------|
| **Hybrid CNN+GRU** | Conv1D(k=5, 64 filters) + Conv1D(k=11, 64 filters) → AdaptiveMaxPool → Concat + BiGRU(64) → FC(128→1) |
| **Simple CNN Baseline** | Conv1D(k=7, 64 filters) → AdaptiveMaxPool → FC(64→1) |

**Training:** Adam optimizer, MSE loss, 15 epochs (Hybrid) / 10 epochs (Baseline), batch size 64, 80/20 train-test split.

**Evaluation metric:** Pearson correlation coefficient between predicted and true binding scores.

---

### Approach 2 — Synthetic Data with Simulated Structure Features

**Notebook:** `che629_project_secondApproach.ipynb`

**Goal:** Explore the effect of adding RNA secondary structure information using 9-feature encoding, still on synthetic data.

**Dataset:**
- 20,000 synthetic sequences of length 40
- Binding score based on **multiple biologically-inspired motifs**: `UGCAUG`, `UUUU`, `UGUG`, `GAGA`, `AUGA`
  ```
  score = Σ(motif counts) × 1.5 + GC_content + N(0, 0.5)
  ```

**Encoding:**
- **4-channel one-hot** (sequence) + **5-channel structure** (random Dirichlet-sampled probabilities simulating structure uncertainty)
- Input shape: `(N, 40, 9)`

**Model (PyTorch):**
- Updated Hybrid CNN+GRU: input channels changed from 4 → 9
- Architecture: Conv1D(9→64, k=5) + Conv1D(9→64, k=11) → Pool → Concat + BiGRU(9→128 bidirectional) → FC(256→64→1)

**Training:** Adam (lr=5e-4), 8 epochs, MSE loss.

**Key limitation:** Structure probabilities were randomly sampled rather than computed from actual RNA folding, making this an approximation used to test the pipeline before moving to real data.

---

### Final Model — Real RNAcompete 2009 Data, Full Reproduction

**Notebook:** `Final_model.ipynb`

**Goal:** Faithful reproduction of the DLPRB paper using real experimental binding data and proper RNA structure predictions.

#### Dataset

- **Source:** `rnacompete2009.h5` — the RNAcompete 2009 benchmark dataset
- **Format:** HDF5; each protein stored under its own group with splits `X_train`, `X_valid`, `X_test`, `Y_train`, etc.
- **Representation:** Sequences encoded as `(N, 9, L)` tensors (transposed to `(N, L, 9)` for Conv1D)

**9-dimensional feature vector per nucleotide position (Section 2.4 of source paper):**

| Dimensions | Feature |
|-----------|---------|
| 0–3 | One-hot sequence encoding: A, G, C, U |
| 4–8 | RNA structure probabilities: hairpin loop, internal loop, multi-loop, external loop, paired |

**Target:** Continuous binding intensity (RNAcompete fluorescence signal). Outliers clipped at the 99.5th percentile before training.

---

#### Architecture 1: CNN (Primary Model)

```
Input (N, L, 9)
    ↓
Conv1D(k=5,  128 filters, ReLU) → GlobalMaxPool1D   ┐
Conv1D(k=11, 128 filters, ReLU) → GlobalMaxPool1D   ┘ → Concatenate (256-dim)
    ↓
Dense(128, ReLU) → Dropout(0.3) → Dense(1)
```

- Multi-scale filters capture short motifs (k=5) and longer contextual patterns (k=11)
- GlobalMaxPool collapses each filter's response to its strongest activation across all positions
- L2 regularization (`λ=1e-4`) applied to convolutional and dense layers

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 1e-4) |
| Loss | Mean Squared Error |
| Batch size | 128 |
| Max epochs | 60 |
| Early stopping patience | 8 epochs (monitors val Pearson r) |
| LR reduction | ×0.5 after 4 epochs of no improvement |
| Total parameters | ~270,000 |

---

#### Architecture 2: Bidirectional GRU (RNN)

```
Input (N, L, 9)
    ↓
BiGRU(64 units per direction) → last hidden state (128-dim)
    ↓
Dense(128, ReLU) → Dropout(0.3) → Dense(1)
```

- GRU chosen over LSTM for lower parameter count with comparable performance (per paper recommendation)
- Bidirectional processing reads the RNA sequence in both 5′→3′ and 3′→5′ directions

---

#### Training Strategy

Three experimental phases were run across all proteins in the dataset:

| Phase | Architecture | Input | Purpose |
|-------|-------------|-------|---------|
| 1 | CNN | Sequence + Structure (9-dim) | Main model |
| 2 | RNN (BiGRU) | Sequence + Structure (9-dim) | Architecture comparison |
| 3 | CNN | Sequence only (4-dim) | Ablation — structure contribution |

Each phase trains and evaluates a separate model per protein, saving `.keras` model files to disk.

---

#### Evaluation

**Primary metric:** Pearson correlation coefficient (*r*) between predicted and true binding intensities on the held-out test set.

**Custom Keras metric:**
```python
def pearson_metric(y_true, y_pred):
    num = Σ (y_true - ȳ_true)(y_pred - ȳ_pred)
    den = √[Σ(y_true - ȳ_true)² · Σ(y_pred - ȳ_pred)²]
    return num / den
```

Results are compiled into a per-protein table and saved as `per_protein_results.csv`, including:
- `CNN (seq+struct)` — Pearson r
- `RNN (seq+struct)` — Pearson r
- `CNN (seq only)` — Pearson r
- `Struct gain (CNN)` — difference showing structure contribution

---

#### Ablation: Value of RNA Structure Information

The impact of the 5 structure channels is quantified by comparing CNN (9-dim) vs CNN (4-dim seq-only):

```
Δr = r(seq + struct) − r(seq only)
```

Statistical significance tested via **Wilcoxon signed-rank test** (one-sided, `struct+seq > seq only`).

> Paper reports an average Δr ≈ +0.014 improvement from adding structure information.

Visualized as:
1. Scatter plot: seq+struct Pearson r vs seq-only Pearson r per protein (diagonal = no gain)
2. Bar chart: per-protein Δr (green = improvement, red = degradation)

---

#### Motif Visualization (Interpretability)

CNN filters are interpreted as learned RNA-binding motifs using **Position Weight Matrix (PWM)** extraction:

1. For a given filter, extract activations on the test set
2. Identify the top 2% most-activated sequences (98th percentile threshold)
3. Collect the subsequence window at the position of maximum activation
4. Normalize frequency counts to obtain a PWM

**Visualization:**
- **Sequence logo** (via `logomaker`) showing nucleotide preferences at each position
- **Structure bar chart** showing probabilities of 5 structural contexts (Paired, Hairpin, Inner-loop, Multi-loop, External) at each position

An **interactive Jupyter widget dashboard** (`ipywidgets`) allows on-the-fly exploration:
- Dropdown selectors for protein and architecture
- Slider for filter index (0–127 = k=5 filters; 128–255 = k=11 filters)
- Slider for activation percentile threshold

---

## Technology Stack

| Component | Approach 1 & 2 | Final Model |
|-----------|---------------|-------------|
| Framework | PyTorch | TensorFlow / Keras |
| Data | Synthetic CSV | HDF5 (RNAcompete 2009) |
| Structure encoding | None / Random Dirichlet | Real structure probabilities |
| Optimizer | Adam | Adam + ReduceLROnPlateau |
| Regularization | None | L2 + Dropout |
| Motif visualization | None | PWM + Logomaker |
| Interpretability | Scatter plots | Filter activation + logos + interactive dashboard |

---

## Key Findings

- **Structure helps:** Including RNA secondary structure probabilities alongside sequence one-hot encoding consistently improves Pearson r, aligning with the paper's reported ~0.014 average gain (confirmed by Wilcoxon test).
- **CNN vs RNN:** Both architectures are competitive; multi-scale CNN filters are particularly effective at capturing discrete binding motifs.
- **Multi-scale filters matter:** Using both short (k=5) and long (k=11) convolutional filters captures motifs at different length scales, mimicking the diversity of RBP recognition sites.
- **Per-protein variation:** Performance varies substantially across RBPs — some proteins are well-predicted by sequence alone, while others rely more heavily on structural context.

---

## File Structure

```
project/
├── che629_project_firstApproach.ipynb   # Approach 1: synthetic, seq-only, PyTorch
├── che629_project_secondApproach.ipynb  # Approach 2: synthetic, seq+struct, PyTorch
├── Final_model.ipynb                    # Final: real data, TF/Keras, full pipeline
├── data/
│   ├── raw/rna.csv                      # Approach 1 synthetic dataset
│   └── real_subset/rna_subset.csv       # Approach 2 synthetic dataset
└── (Google Drive)
    ├── rnacompete2009.h5                # Real benchmark dataset
    ├── saved_models/                    # Per-protein .keras model files
    └── results/
        ├── per_protein_results.csv      # Summary table
        ├── training_curves_cnn.png      # Loss/Pearson curves per protein
        ├── scatter_HuR.png              # Predicted vs actual for HuR
        ├── ablation_structure.png       # Structure contribution analysis
        └── motif_*.png                  # Filter PWM visualizations
```

---

## References

- Alipanahi, B. et al. (2015). *Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning.* Nature Biotechnology.
- Ray, D. et al. (2009). *Rapid and systematic analysis of the RNA recognition specificities of RNA-binding proteins.* Nature Biotechnology. (RNAcompete 2009 dataset)

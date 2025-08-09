# T3former: Temporal Graph Classification with Topological Machine Learning

T3Former is a lightweight and efficient temporal graph classification model that integrates topological, spectral, and structural features through Transformer encoders and self-attention. It avoids traditional GNN-RNN stacks by leveraging temporal graph properties in a more compact and scalable manner.

## 🔍 Overview

Our model processes temporal graphs using a sliding window approach. From each window, it extracts structural and topological summaries, transforms them via attention-based encoders, and classifies entire sequences using fused representations.

### Key Features

- **Sliding Window Temporal Modeling**:  
  - Window length: δ = 6  
  - Stride: σ = 4  
  - Number of windows per graph:   
**N = floor((t_max − t_min − δ) / σ) + 1**


- **Feature Extraction per Window**:
  - **Topological Tokens**:  
    - Betti-0 and Betti-1 from the clique complex (persistent homology).
    - Passed through a **Transformer encoder** to model temporal consistency.
  - **Spectral Tokens**:  
    - Degree of Spectrality (DoS) histogram with 4 bins using the normalized Laplacian spectrum.
    - Also passed through a **Transformer encoder** for sequence modeling.
  - **Global Encoder**:  
    - A 2-layer GraphSAGE encoder applied to the static graph using either:
      - Temporal degree features, or  
      - Domain-specific node features.

- **Fusion and Classification**:
  - The outputs from the topological and spectral Transformer encoders are fused with the global structural embedding using a **self-attention mechanism**.
  - A final **linear layer** performs classification.

---

## 📊 Results & Efficiency

T3Former achieves high accuracy across various temporal graph classification datasets while being significantly more efficient (15 times faster) than GCN+LSTM-based models.

See **Appendix Table 7** in the paper for a detailed runtime breakdown.

---

## 📁 Dataset Links

You can download the datasets used in this project from the following sources:

- **[NeuroGraph Dataset ](https://neurograph.readthedocs.io/en/latest/)**
- **[Social Network Dataset ](https://chrsmrrs.github.io/datasets/docs/datasets/)**
- **[Traffic Dataset ](https://torch-spatiotemporal.readthedocs.io/en/latest/modules/datasets_in_tsl.html)**

---

## 🛠️ Requirements

```bash
python >= 3.8  
torch >= 1.10  
torch-geometric >= 2.0   
numpy  
scipy  

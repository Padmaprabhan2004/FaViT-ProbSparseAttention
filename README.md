# FaViT-ProbSparse: Factorized Spatio-Temporal Self-Attention with ProbSparse Sampling 


## Overview

This repository contains the experiments and ablations on *Factorized Spatio-temporal Self-Attention mechanisms and Position Encodings for improved correlation structure and long-dependency modelling*.  

The work builds upon recent transformer architectures such as **FaViT** (Factorized Vision Transformer, Qin et al., 2023) and **Informer** (Zhou et al., 2021), with a specific focus on:

- **ProbSparse Attention** for efficient cross-window fusion  
- **Factorized self-attention (FaSA)** for mixed-grained multi-head representation  
- **Legendre Polynomial-based Positional Encodings (PoPE)** for improved correlation structure  
- **ΔV-based Tangential Space Attention** to capture implicit token order  


## Key Contributions

1. **ProbSparse Aggregation in FaViT:**
   - Replaced standard pooling (Global Avg/Max) with a *ProbSparse* key-query selection mechanism.
   - Achieved **+1.87% Top-1 accuracy** on CIFAR-100 compared to MaxPooling baseline.
   - Uses KL-divergence-based query sparsification to retain high-attention tokens:
     \[
     D_{KL}(P_i \| U) = \log L_k + \sum_j P_i(j)\log P_i(j)
     \]
   - Selects top-*L2* keys and top-*log(Lq)* queries for final attention computation.

2. **Cross-Term Analysis in Self-Attention:**
   - Expanded self-attention correlations into token-token, token-pos, pos-token, and pos-pos terms.
   - Demonstrated that **cross-terms (token↔pos)** are essential for semantic ordering.

3. **Positional Encoding Studies:**
   - **Legendre Polynomial Embeddings (PoPE):** Orthogonal polynomial basis improving temporal correlation modeling.  
   - **ΔV Attention:** Momentum-based encoding that implicitly captures sequence order without explicit positions.

4. **Unified Interpretation:**  
   - ProbSparse fusion acts as a *data-driven pooling operator*, balancing efficiency and contextual richness across attention heads.

---

## Datasets and Experimental Setup

| Dataset | Task | Model | Notes |
|----------|------|--------|-------|
| CIFAR-100 | Image classification | FaViT-B0/B2 | Cross-window fusion ablation |
| ImageNet-1K | Large-scale classification | FaViT-B0/B2 | High-res validation of ProbSparse |
| ETTh1 / ETTm1 | Multivariate time-series forecasting | Informer | Position encoding & ΔV analysis |
 

## Tools & Frameworks

- **Framework:** PyTorch 2.0.1 + CUDA 11.8  
- **Hardware:**  
  - NVIDIA **T4 GPU** for time-series tasks (ETTh1, ETTm1)  
  - **Vikram-1000 HPC Cluster (A100 GPU)** for CIFAR-100/ImageNet1K training  
- **Optimization:**  
  - Optimizer: AdamW  
  - Scheduler: Cosine decay with warm-up/cool-down  
  - Weight Decay: 0.05  
  - Batch size: 128  
  - Learning rate: 1e-4  

---

## Results Summary

### CIFAR-100 (FaViT-B0 Ablation)

| Aggregator | Params (M) | FLOPs (G) | Top-1 Acc (%) |
|-------------|-------------|-----------|----------------|
| Pointwise Conv | 3.5 | 0.6 | 68.6 |
| Linear Layer | 3.5 | 0.6 | 67.3 |
| Global Avg Pool | 3.4 | 0.6 | 68.6 |
| Max Pool (orig) | 3.4 | 0.6 | 68.9 |
| **ProbSparse (Ours)** | 3.4 | 0.6 | **70.77** |





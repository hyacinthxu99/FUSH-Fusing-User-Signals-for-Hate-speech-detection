# FUSH: Fusing User Signals for Hate Speech Detection

This repository contains the official implementation and dataset for the paper:

**"Improving Hate Speech Detection by Fusing Textual and User Interaction Representations in Online Communities"** 
---


## Overview

Detecting hate speech in online communities is challenging due to its implicit and context-dependent nature. This project introduces **FUSH**, a framework that enhances hate speech detection by jointly modeling:

- Textual semantics (via RoBERTa)
- User interaction signals (via heterogeneous graph modeling)

By integrating both modalities, FUSH achieves significantly better performance than text-only baselines.

---

## Repository Structure
├── dataset/ # Dataset files

└── model/ # Model implementation (FUSH)




## Label Definition

The dataset uses **binary classification labels**:

- `0` → Normal (non-hate speech)
- `1` → Hate speech

---

## TODO

- [ ] We are planning to further annotate hate speech instances with **fine-grained target categories**, specifying the exact target of hate.

---

## License

This dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You are free to use, share, and adapt the data for **non-commercial research purposes**, provided that proper attribution is given.

Commercial use is strictly prohibited.

For more details, see:
https://creativecommons.org/licenses/by-nc/4.0/
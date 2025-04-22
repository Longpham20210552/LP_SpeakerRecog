# Speaker Recognition using Lightweight Deep Reshape Dimensions Net - Training from scratch 

A deep learning-based speaker recognition system with **spoofing detection** using a scaled-down version of ReDimNet.

## Project Overview

This project aims to build an **efficient and compact speaker recognition model** for both **closed-set** and **open-set** scenarios, with **robust spoofing detection** capability.

Key contributions:
- Scaled down ReDimNet to a **580k parameter lightweight model**.
- Implemented a **cross-validation training pipeline** for fair model evaluation.
- Developed and tested **two training algorithms**: standard classification and SRPL-based.
- Achieved **97% accuracy on closed-set**, and **91.5% accuracy on open-set + spoofing detection**.

## Technical Details

- **Model Architecture**: Small ReDimNet (Deep Reshape Dimensions Network)
- **Spoofing Detection**: Unknown Labeling & SRPL (Speaker Reciprocal Points Learning)
- **Embedding Space**: Optimized for class separation and adversarial robustness
- **Evaluation Protocols**: Cross-validation on clean/noisy/synthetic data
- **Training Losses**:
  - Classification (CrossEntropy / AAMSoftmax)
  - Metric Learning (SRPL Loss)


![image](https://github.com/user-attachments/assets/bd4a455e-ae3b-49c2-aa94-c83aeda43509)

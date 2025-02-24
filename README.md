
# Inferring Cosmological Parameters from Void Properties Using Deep Learning


## Overview

This repository contains code and models for inferring cosmological parameters from cosmic void properties using **fully connected neural networks**. Our approach is based on machine learning techniques applied to large-scale simulated void datasets. The project builds on the paper *Machine-learning Cosmology from Void Properties* (Wang et al. 2023) and utilizes the **GIGANTES** dataset, which contains over a billion cosmic voids.

## Key Features

- **Dataset:** Uses 2000 void catalogs from the GIGANTES dataset at redshift *z = 0.0*
- **Machine Learning Model:** Fully connected neural network with **four dense layers (100 nodes each, ReLU activation)**
- **Input Features:** Void ellipticity, density contrast, and radius
- **Target Outputs:** Cosmological parameters:
  - **Ωm** (matter density)
  - **σ8** (amplitude of density fluctuations)
  - **ns** (spectral index of primordial fluctuations)
- **Optimization:** AdamW optimizer with weight decay to prevent overfitting
- **Performance Metrics:** Evaluated using R², RMSE, and MAE

## Dataset & Preprocessing

The **GIGANTES** dataset was processed by extracting key void properties:
- **Ellipticity (ϵ)**: Derived from the void shape data
- **Radius (r)**: Obtained from void center data
- **Density contrast (Δ)**: Derived from density measurements

These features were binned into histograms (18 bins each) to serve as input for the neural network.

## Model Architecture

We trained three different neural network models:

| Model     | Input Features | Output                          | Loss Function | Optimizer |
|-----------|--------------|--------------------------------|--------------|------------|
| **Model 1** | ϵ (void ellipticity histogram) | Ωm | Mean Squared Log Error (MSLE) | AdamW |
| **Model 2** | (ϵ, r, Δ) concatenated | Ωm | MSLE | AdamW |
| **Model 3** | (ϵ, r, Δ) concatenated | (Ωm, Ωb, h, ns, σ8) | Custom loss (MSE + Log Error) | AdamW |

Hyperparameters were optimized using **Optuna**, improving model performance by selecting the best learning rates, batch sizes, and weight decay values.

## Training & Evaluation

- **Training:** 80/20 train-test split, batch size = 128, training epochs = 500-1000
- **Evaluation Metrics:**
  - **R² Scores:**
    - **Model 1 (Ωm):** 0.6852
    - **Model 2 (Ωm):** 0.7534
    - **Model 2 (σ8):** 0.7564
    - **Model 3 (Ωm):** 0.7582
    - **Model 3 (σ8):** 0.7612
  - **Other Metrics:** RMSE and MAE confirm accuracy improvements through hyperparameter tuning

## Results

- **Strong Predictive Performance for Ωm and σ8:**
  - Model 3, trained on all void properties, achieved the highest accuracy for these parameters
- **Limited Performance for ns:**
  - Model struggled to accurately infer ns, similar to the findings in Wang et al. (2023)
- **Feature Importance:**
  - Adding radius and density contrast features improved performance over using ellipticity alone

## Future Improvements

- **Expand feature set:** Incorporate additional void properties (e.g., higher-order shape descriptors)
- **Explore alternative architectures:** Use CNNs or transformers for hierarchical feature extraction
- **Cross-validation on observational data:** Validate performance beyond simulated datasets

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib
- Optuna (for hyperparameter tuning)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-repo/cosmo-void-ml.git
cd cosmo-void-ml
pip install -r requirements.txt

# 🛒 Amazon ML Challenge 2025: Multimodal price prediction system

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-blue)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Project Overview

This repository contains a high-performance **Multimodal Late-Fusion Stacking Ensemble** developed for the **Amazon ML Challenge 2025**. The objective is to predict the optimal price of e-commerce products using a combination of textual catalog metadata and raw product images, strictly adhering to an 8-Billion parameter limit and optimizing for the **Symmetric Mean Absolute Percentage Error (SMAPE)** metric.

The solution effectively processes a dataset of 150,000 products by fusing high-dimensional sparse text representations with dense visual embeddings, ultimately bridging the gap between semantic marketing descriptions and visual product quality.

---

##  System Architecture

The pipeline is designed to handle massive multi-modal datasets efficiently, utilizing custom multi-GPU processing and out-of-core memory management.

### 1. Data Engineering & Feature Extraction (Dual-Stream)
* **Text Stream (NLP):**
    * **Domain-Specific Parsing:** Explicit extraction of `item_name`, `product_description`, and `bullet_points`.
    * **Brand NER:** Utilized `spaCy` Named Entity Recognition combined with custom regex and a blacklist to isolate high-impact brand names.
    * **IPQ Extraction:** Aggressive regex patterns to extract Item Pack Quantity (e.g., distinguishing "Pack of 1" from "Pack of 12").
    * **Vectorization:** Fused TF-IDF (word/ngram), Character N-grams, and Hashing Vectorizer to generate a 1.27M+ dimensional sparse matrix.
* **Vision Stream (CV):**
    * **Pipelined CLIP Processing:** Deployed OpenAI's `clip-ViT-B-32` using a custom Producer-Consumer architecture. CPU workers handle parallel image downloading/decoding, while multi-GPU workers process batches (size 256) to generate 512-dimensional visual embeddings.
    * **Resilient Chunking:** Embedded a resumable checkpointing system to survive platform timeouts during the processing of 150k images.

### 2. Level 0 Base Learners (The Fleet)
To ensure maximum ensemble diversity, the architecture utilizes three distinct modeling paradigms:
* **High-Dimensional Sparse Models (Text-Heavy):** LightGBM and XGBoost trained with **Custom SMAPE Gradients/Hessians** alongside fast linear models (Ridge, SGD) and Factorization Machines (FFM).
* **Dense SVD Models (Concept-Heavy):** `Faiss` (Approximate Nearest Neighbors) acting as a non-parametric target encoder, combined with a Keras MLP trained on Truncated SVD features.
* **End-to-End Vision:** A fine-tuned Swin Transformer (`swin_base_patch4_window7_224`) processing raw images to predict log-price directly.

### 3. Meta-Model (Level 1 Stacking) & Post-Processing
* **Stacking:** A `RidgeCV` meta-model learns to blend the Level 0 predictions. It utilizes engineered meta-features (differences and ratios between tree-based text predictions and deep learning visual predictions) to resolve inter-model disagreements.
* **Calibration:** Isotonic Regression is applied to map the predicted price distribution to the historical training distribution.
* **Target Transformation:** Predictions are mapped using `np.log1p` during training and reversed with `np.expm1` to handle the extreme right-skew of e-commerce pricing.

---

## 🛠️ Tech Stack & Dependencies

* **Core:** Python, Pandas, NumPy, SciPy
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Vowpal Wabbit
* **Deep Learning & Vision:** PyTorch, HuggingFace Transformers, SentenceTransformers (`CLIP`), Keras
* **NLP:** NLTK, spaCy (`en_core_web_sm`)
* **Optimization:** Faiss, multiprocessing

---

## Model Evaluation & Performance

### Primary Metric: SMAPE
The primary evaluation metric for this challenge is the **Symmetric Mean Absolute Percentage Error (SMAPE)**. SMAPE measures the relative accuracy of the predicted prices, bounding the error between 0% and 200%. 

**Formula:**
`SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|) / 2) * 100%`

### Optimization Strategy
Because standard regression metrics (like RMSE or MAE) struggle with the relative nature of SMAPE, this pipeline employs specific techniques to optimize the target directly:
* **Custom Loss Functions:** The tree-based base learners (XGBoost and LightGBM) were trained using custom-written, GPU-accelerated gradients and Hessians to minimize an approximation of SMAPE directly.
* **Target Transformation:** E-commerce prices are heavily right-skewed. The models are trained on `log1p(price)` to stabilize variance, and the final predictions are reversed using `expm1(price)`.
* **Distribution Calibration:** Final predictions are passed through an Isotonic Regression calibrator to ensure the output price distribution perfectly matches the historical training data distribution.

### Results
* **Local Validation (SMAPE):** 47.9%
* **Public Leaderboard (SMAPE):** 47.48%
* **Final Ranking:** 510/82803 teams(Top 0.6%)
  
## Contributing

I welcome contributions to this project! Whether it's reporting a bug, suggesting a new feature, improving documentation, or submitting code, your help is highly appreciated.


---


## Contact

For any questions or collaborations, please reach out at amansinghbudhala15@gmail.com

---



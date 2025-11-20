# Flexible ML Predictor

## Description

An optimized, end-to-end **Auto-ML command-line tool** designed for the comprehensive analysis of tabular data (CSV, Excel, JSON). This project bridges the gap between traditional and advanced machine learning by integrating **Supervised Learning** (Regression/Classification), **Unsupervised Learning** (Clustering/PCA), and **Reinforcement Learning** (Trend Prediction) into a single, memory-efficient pipeline.

It guarantees model integrity using **scikit-learn Pipelines** to prevent data leakage, maximizes speed with **RandomizedSearchCV**, and provides deep interpretability through **feature importance** reporting.

---

## Key Features

### Machine Learning Modules
* **Supervised Learning:** Automatically detects and trains **Regression** (numeric target) or **Classification** (categorical target) models using robust algorithms (Random Forest, Gradient Boosting, Linear/Logistic Regression).
* **Unsupervised Analysis:** In-tool options for **K-Means Clustering** and **Principal Component Analysis (PCA)** to discover hidden patterns and reduce dimensionality.
* **Reinforcement Learning (NEW):** Includes a custom **Gymnasium Environment** that treats data as a time-series. Trains a **PPO Agent** (via Stable Baselines3) to predict future trends (Increase/Decrease) based on sequential state features.

### Engineering & Performance
* **Data Leakage Prevention:** Strictly isolates preprocessing (scaling, imputation) within a `ColumnTransformer` to ensure test data remains unseen during training.
* **Memory Optimized:** Automates **sparse encoding** for categoricals and **numeric downcasting** to handle large datasets efficiently.
* **Smart Feature Engineering:** Automatically detects datetime columns and extracts predictive features (**month, day, hour**).
* **Fast Model Selection:** Uses **RandomizedSearchCV** to efficiently find the best hyperparameters without the computational cost of exhaustive grid searches.

### Utilities
* **Model Persistence:** Save and load full pipelines (preprocessor + model) using `joblib` and RL agents using `.zip` format.
* **Deep Analysis:** Efficient **Pandas-based querying** and visualization (histograms, scatter plots).

---

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/flexible-ml-predictor.git
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    *Includes `stable-baselines3` and `gymnasium` for RL support.*
    
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Run the main application from your terminal to access the interactive menu:

```bash
python main.py
```
---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
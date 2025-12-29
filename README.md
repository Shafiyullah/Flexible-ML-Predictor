# Predicto-AI

## Description

An advanced, dual-interface **Auto-ML platform** designed for the comprehensive analysis of tabular data. This project now offers both a robust **Command-Line Interface (CLI)** and a professional **Streamlit Web Dashboard**.

It bridges the gap between traditional and modern AI by integrating:
*   **Supervised Learning:** Regression & Classification (with **XGBoost** & **HistGradientBoosting** support).
*   **Time-Series Forecasting:** Powered by **Facebook Prophet**.
*   **Reinforcement Learning:** Trend prediction using **PPO Agents**.
*   **Unsupervised Learning:** Clustering & PCA.

Engineered for performance, it features **PyArrow**-accelerated data loading, compressed model artifacts, and strict data leakage prevention using scikit-learn Pipelines.

---

## Key Features

### 🖥️ Interactive Dashboard
A professional web interface built with **Streamlit** offering:
*   **Modern Dark UI:** A sleek, glassmorphism-inspired dark theme designed for professional use.
*   **Robust Architecture:** Features a "Safe Boot" system to handle heavy ML dependency loading without crashing.
*   **Data Upload:** Drag-and-drop CSV/Excel/JSON files with instant **PyArrow** processing.
*   **EDA:** Tabbed interactions for distributions, correlations, and feature overviews using **Plotly**.
*   **No-Code Training:** Train XGBoost/Sklearn models with one click.
*   **Interactive Forecasting:** Visual time-series prediction with dynamic zoom/pan.

### 🤖 Advanced Algorithms
*   **Gradient Boosting Powerhouse:** Integrated **XGBoost** and **HistGradientBoosting** (Regressor/Classifier) alongside Random Forest for state-of-the-art performance on datasets of any size.
*   **Prophet Forecasting:** specialized additive models for accurate time-series prediction, handling seasonality and trends automatically.
*   **RL Trend Predictor:** Custom **Gymnasium Environment** trained with **Stable Baselines3 (PPO)** to predict future market/data directions.

### ⚙️ Engineering & Optimization
*   **High-Performance I/O:** Uses `engine='pyarrow'` for ultra-fast CSV reading and memory efficiency.
*   **Storage Optimization:** Models are saved with **level-3 compression** to reduce artifact size without losing accuracy.
*   **Vectorized Processing:** Replaces iterative loops with **vectorized pandas operations** for ultra-fast string cleaning and type inference.
*   **Memory Efficiency:** Smart downcasting of numeric columns (Float64 -> Float32/16) to reduce memory footprint by up to 50%.
*   **Robust Pipelines:** Automates scaling, imputation, and encoding within `searchable` pipelines to prevent data leakage.
*   **Production Standards:** Fully PEP8 compliant with comprehensive **Google-Style Docstrings** and structured logging.

---

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Shafiyullah/Predicto-AI.git
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    *   **Windows:** `venv\Scripts\activate`
    *   **Linux/Mac:** `source venv/bin/activate`

3.  **Install Dependencies:**
    *Includes `xgboost`, `prophet`, `streamlit`, and `plotly`.*
    
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Option 1: Web Dashboard (Recommended)
Launch the full GUI experience:
```bash
streamlit run dashboard.py
```
*Access the dashboard in your browser at `http://localhost:8501`*

### Option 2: Command Line Interface
Run the classic terminal-based tool:
```bash
python main.py
```
---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
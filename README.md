# Multi-Output Time Series Forecasting: NIFTY-50 (Tata Steel)

## Project Overview
This project implements a Deep Learning approach to forecast the stock market behavior of **Tata Steel** using historical data from the National Stock Exchange (NSE) India (2000 - 2021). 

The goal is to perform **Multi-Output Forecasting**, where the model takes a window of historical data (all 11 features) to predict the next 5 days of values for those same 11 features simultaneously.

### Objectives
*   Implement a **Recurrent Neural Network (LSTM)** using PyTorch.
*   Process and clean high-variance financial data.
*   Compare model performance across two scenarios:
    1.  **Case 1:** 5-day input window -> 5-day prediction.
    2.  **Case 2:** 10-day input window -> 5-day prediction.

---

## Data Pipeline & Methodology

### 1. Data Preprocessing & EDA
*   **Feature Engineering:** Dropped non-informative categorical columns (`Symbol`, `Series`).
*   **Handling Missing Values:** Addressed the digital tracking gap in `Trades` and `Deliverable Volume` (NSE tracking began post-2011) by using zero-imputation to preserve 11 years of valuable price history without introducing "mean-flatline" bias.
*   **Non-Linear Scaling:** Performed **Log Transformations** (`np.log1p`) on highly skewed features (Volume, Turnover, Trades) to stabilize variance and improve LSTM convergence.
*   **Normalization:** Applied `MinMaxScaler` fitted exclusively on training data to prevent data leakage.

### 2. Model Architecture
Built a custom **Multi-Output LSTM** class in PyTorch:
*   **Input Layer:** 11 features.
*   **Hidden Layers:** 2 stacked LSTM layers with 64 hidden units.
*   **Output Layer:** Linear layer mapping to a 55-dimension vector, reshaped to `(5 days, 11 features)`.
*   **Loss Function:** Mean Squared Error (MSE).
*   **Optimizer:** Adam Optimizer (learning rate = 0.001).

---

## Performance Metrics
The model was evaluated using **MSE, RMSE, and MAE**.

| Metric | Value (Log-Scaled Space) |
| :--- | :--- |
| **Test MSE** | ~0.0041 |
| **Average RMSE** | 19.77 (Price Features) |

*The model demonstrated high accuracy in tracking price trends (Close, VWAP) while maintaining a conservative "central tendency" for more volatile volume metrics.*

---

## Visualizations
The project includes comprehensive plotting:
*   **Learning Curves:** Training vs. Validation loss to identify the optimal "Early Stopping" point (Epoch 30).
*   **Feature Grid:** A 6x2 grid visualizing 1-day ahead predictions for every single feature against true values.
*   **Case Comparison:** Side-by-side comparison of 5-day vs. 10-day input windows.

---

## Tech Stack
*   **Language:** Python 3.x
*   **Deep Learning:** PyTorch
*   **Data Analysis:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-learn

---

## How to Run
1. Clone the repository:
   ```bash
   git clone (https://github.com/msh165/jmi-msc-ai-ml-RNN-tatasteel-multi-feature-output.git)
2. Install Dependencies
   ```bash
   pip install torch pandas matplotlib scikit-learn seaborn
3. Open tatasteel_analysis.ipynb in Jupyter Notebook or VS Code and run all cells.

# Predicting Customer Churn in Telecommunications

This project focuses on analyzing and predicting customer churn in the telecommunications industry using machine learning techniques. The goal is to build models that can identify customers at high risk of leaving, allowing for proactive retention strategies. Two primary models are developed and compared: Logistic Regression and XGBoost.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Features](#features)
4.  [Methodology](#methodology)
5.  [Requirements](#requirements)
6.  [Setup and Installation](#setup-and-installation)
7.  [Usage](#usage)
8.  [Output](#output)
9.  [Interpreting Results](#interpreting-results)
10. [File Structure](#file-structure)
11. [Presentation](#presentation)
12. [License](#license)

## Project Overview

Customer churn is a critical metric for subscription-based businesses like telecommunications. This project aims to:
*   Preprocess and prepare a real-world telecommunications dataset.
*   Implement and train two supervised classification models:
    *   Logistic Regression (as an interpretable baseline)
    *   XGBoost (a powerful gradient boosting algorithm)
*   Evaluate and compare the performance of these models using various metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC).
*   Identify key features and customer attributes that are most indicative of churn.
*   Address moderate class imbalance present in the dataset.

## Dataset

The project utilizes the publicly available **"Telco Customer Churn"** dataset from IBM Watson Analytics.
*   **Source:** [Often found on Kaggle or IBM's site, e.g., [Kaggle Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
*   **Observations:** Approximately 7,043 customers.
*   **Features:** Around 20 features detailing customer demographics, subscribed services, contract details, billing information, and churn status.
*   **Target Variable:** `Churn` (Yes/No), indicating whether the customer left within the last month.
*   **Class Imbalance:** The dataset exhibits a moderate class imbalance, with approximately 26.5% of customers having churned.

## Features

The script `project.py` performs the following key functions:
*   **Data Loading:** Loads the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset.
*   **Data Cleaning & Preprocessing:**
    *   Handles missing values in `TotalCharges`.
    *   Converts categorical features to numerical using One-Hot Encoding.
    *   Scales numerical features using StandardScaler.
    *   Maps the target variable 'Churn' to binary (0/1).
*   **Model Training:**
    *   Splits data into training (80%) and testing (20%) sets, stratified by the target variable.
    *   Trains a Logistic Regression model (using `class_weight='balanced'` for imbalance).
    *   Trains an XGBoost model (using `scale_pos_weight` for imbalance).
*   **Model Evaluation:**
    *   Calculates and prints Accuracy, Precision, Recall, F1-Score, and ROC AUC for both models on the test set.
    *   Generates and displays/saves Confusion Matrices.
*   **Results Visualization:**
    *   Plots and saves ROC Curves for both models.
    *   Plots and saves Confusion Matrices.
    *   Plots and saves top feature importances from XGBoost.
    *   Plots and saves top coefficients from Logistic Regression.
    *   Plots and saves the churn distribution pie chart.
*   **Output Generation:**
    *   Saves visual plots as `.png` files.
    *   Saves a summary of model performance metrics as `model_performance_summary.csv`.

## Requirements

*   Python 3.8+
*   The following Python libraries (can be installed via pip):
    *   `pandas`
    *   `numpy`
    *   `matplotlib`
    *   `seaborn`
    *   `scikit-learn`
    *   `xgboost`

## Setup and Installation

1.  **Clone the repository (if applicable) or download the files.**
    ```bash
    # If it's a git repository
    # git clone <repository-url>
    # cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    You can create a `requirements.txt` file with the following content:
    ```
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    xgboost
    ```
    Then install using:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install them individually:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```

4.  **Place the Dataset:**
    Download the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file and ensure it is in the same directory as the `project.py` script, or update the path in the script accordingly.

## Usage

To run the project, navigate to the project directory in your terminal and execute the Python script:

```bash
python project.py

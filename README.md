# Statistical Inference on Cardiovascular Risk: Framingham Heart Study Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

This project presents a comprehensive statistical inference analysis of the renowned **Framingham Heart Study** dataset. The primary objective is to identify and quantify the key risk factors associated with the 10-year risk of coronary heart disease (CHD).

By employing rigorous statistical methods—ranging from hypothesis testing to multivariate regression—this study isolates the physiological, behavioral, and demographic determinants of cardiovascular health, providing actionable insights for early detection and prevention.

## 📂 Dataset

* **Source:** [Kaggle - Framingham Heart Study Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset/data)
* **Description:** A longitudinal cohort study of ~4,240 residents in Framingham, Massachusetts.
* **Target Variable:** `TenYearCHD` (Binary: 0 = No CHD, 1 = CHD within 10 years).
* **Key Predictors:** Age, Systolic BP, Cholesterol, BMI, Smoking Status, Glucose, Education.

## ⚙️ Methodology

The analysis is structured into four distinct phases:

1.  **Exploratory Data Analysis (EDA):**
    * Visualized distributions of demographic and health metrics.
    * Assessed class imbalance and multicollinearity (e.g., sysBP vs. diaBP).
2.  **Statistical Estimation:**
    * Imputed missing values for Glucose and Education.
    * Calculated Maximum Likelihood Estimates (MLE) for blood pressure.
    * Constructed 95% Confidence Intervals for population cholesterol levels.
3.  **Hypothesis Testing:**
    * **T-Test:** Investigated the impact of smoking on heart rate.
    * **Mann-Whitney U:** Compared glucose levels between diabetic and non-diabetic groups.
    * **ANOVA:** Analyzed the relationship between Education level and Obesity (BMI).
4.  **Regression Modeling:**
    * **Logistic Regression:** Built a predictive model for 10-year CHD risk.
    * **Lasso Regression:** Performed feature selection to identify determinants of Systolic Blood Pressure.

## 📊 Key Findings

* **Primary Risk Factors:** **Age** and **Systolic Blood Pressure** are the most dominant predictors of CHD risk.
* **Behavioral Impact:** Smoking is significantly associated with elevated heart rate ($p < 0.001$) and acts as a dose-dependent risk factor for CHD.
* **Socioeconomic Link:** Lower educational attainment is statistically linked to higher BMI ($p < 0.001$), highlighting a socioeconomic gradient in obesity risk.
* **Diabetes Marker:** Glucose levels differ profoundly between diabetics and non-diabetics ($p \approx 0$), validating it as a critical biomarker.

## 🛠️ Technologies Used

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Statistical Analysis:** SciPy, Statsmodels
* **Machine Learning:** Scikit-learn (Logistic Regression, Lasso, Standardization)

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/abolfazlghandi/framingham-chd-risk-analysis.git](https://github.com/abolfazlghandi/framingham-chd-risk-analysis.git)
    cd framingham-chd-risk-analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
    ```

3.  **Run the Python script:**
    ```bash
    python main.py
    ```

## 📁 Project Structure
```text
├── data/
│   └── framingham.csv          # Raw dataset
├── images/                     # Generated plots
├── src/
│   ├── data_cleaning.py        # Imputation and preprocessing
│   ├── visualization.py        # Plotting functions
│   └── inference.py            # Hypothesis tests and regression models
├── proposal.pdf                # Project Proposal Document
├── report.pdf                  # Final Statistical Report (LaTeX compiled)
├── presentation.pptx           # Summary slides
├── main.py                     # Main script to run the full analysis
└── README.md                   # Project documentation
```
## 🤝 Credits

* **Author:** Abolfazl Ghandi
* **Acknowledgments:**
    * The Framingham Heart Study for the original data collection.
    * Kaggle for hosting the dataset.

---
*This project was conducted as part of a Master's degree portfolio in Data Science & Statistical Inference.*

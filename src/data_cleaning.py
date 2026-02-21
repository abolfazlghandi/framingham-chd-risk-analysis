import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None

def clean_data(df):
    """
    Performs data cleaning and imputation.
    - Imputes missing 'glucose' values with the mean.
    - Imputes missing 'education' values with the mode.
    - Fills other numeric missing values with the mean for regression stability.
    """
    if df is None:
        return None

    print("\n--- Starting Data Cleaning ---")
    
    # Check for initial missing values
    missing_glucose = df['glucose'].isnull().sum()
    missing_education = df['education'].isnull().sum()
    print(f"Initial missing glucose: {missing_glucose}")
    print(f"Initial missing education: {missing_education}")

    # Impute glucose with mean
    glucose_mean = df['glucose'].mean()
    df['glucose'].fillna(glucose_mean, inplace=True)
    print(f"Imputed missing 'glucose' with mean: {glucose_mean:.2f}")

    # Impute education with mode
    education_mode = df['education'].mode()[0]
    df['education'].fillna(education_mode, inplace=True)
    print(f"Imputed missing 'education' with mode: {education_mode}")
    
    # General imputation for other columns (e.g. cigsPerDay, BPMeds, totChol, BMI, heartRate)
    # This is often necessary for scikit-learn models which cannot handle NaNs.
    # We will use mean imputation for remaining continuous variables.
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and df[col].isnull().sum() > 0:
             col_mean = df[col].mean()
             df[col].fillna(col_mean, inplace=True)
             print(f"Imputed remaining missing values in '{col}' with mean: {col_mean:.2f}")

    print("Data cleaning complete.\n")
    return df
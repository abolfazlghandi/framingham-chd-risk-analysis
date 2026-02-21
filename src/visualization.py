import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the images directory exists
if not os.path.exists('images'):
    os.makedirs('images')

def set_style():
    """Sets the seaborn style for plots."""
    sns.set_theme(style="whitegrid")

def plot_distributions(df):
    """
    Generates distributions for Age, BMI, Education, and TenYearCHD.
    Saves the figure as 'images/distributions.png'.
    """
    print("Generating distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Age Distribution
    sns.histplot(df['age'].dropna(), kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribution of Age')

    # BMI Distribution
    sns.histplot(df['BMI'].dropna(), kde=True, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Distribution of BMI')

    # Education Bar Chart
    edu_counts = df['education'].value_counts().sort_index()
    sns.barplot(x=edu_counts.index, y=edu_counts.values, ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Frequency of Education Levels')
    axes[1, 0].set_xlabel('Education')

    # TenYearCHD Bar Chart
    chd_counts = df['TenYearCHD'].value_counts().sort_index()
    sns.barplot(x=chd_counts.index, y=chd_counts.values, ax=axes[1, 1], palette='magma')
    axes[1, 1].set_title('Ten Year CHD Outcome Count')
    axes[1, 1].set_xlabel('TenYearCHD (0=No, 1=Yes)')

    plt.tight_layout()
    plt.savefig('images/distributions.png')
    plt.close()
    print("Saved 'images/distributions.png'")

def plot_relationships(df):
    """
    Generates scatter plot for sysBP vs diaBP and box plot for sysBP by Smoker.
    Saves the figure as 'images/relationships.png'.
    """
    print("Generating relationship plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # sysBP vs diaBP
    sns.scatterplot(data=df, x='sysBP', y='diaBP', alpha=0.5, ax=axes[0], color='teal')
    axes[0].set_title('sysBP vs. diaBP')

    # sysBP by Smoking Status
    sns.boxplot(data=df, x='currentSmoker', y='sysBP', ax=axes[1], palette='pastel')
    axes[1].set_title('sysBP by Smoking Status')
    axes[1].set_xlabel('Current Smoker (0=No, 1=Yes)')

    plt.tight_layout()
    plt.savefig('images/relationships.png')
    plt.close()
    print("Saved 'images/relationships.png'")

def plot_multivariate(df):
    """
    Generates a pair plot for Age, BMI, totChol, Glucose colored by TenYearCHD.
    Saves the figure as 'images/multivariate_pairplot.png'.
    """
    print("Generating multivariate pair plot...")
    pair_vars = ['age', 'BMI', 'totChol', 'glucose', 'TenYearCHD']
    # Drop NaNs locally for plotting to avoid errors
    sns_pair = sns.pairplot(df[pair_vars].dropna(), hue='TenYearCHD', palette='husl', diag_kind='kde')
    sns_pair.savefig('images/multivariate_pairplot.png')
    plt.close()
    print("Saved 'images/multivariate_pairplot.png'")

def plot_correlation(df):
    """
    Generates a heatmap of the correlation matrix.
    Saves the figure as 'images/correlation_heatmap.png'.
    """
    print("Generating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png')
    plt.close()
    print("Saved 'images/correlation_heatmap.png'")

def plot_bmi_fit(bmi_data, mean, std):
    """
    Plots the BMI distribution with a normal curve overlay.
    Saves as 'images/bmi_distribution_fit.png'.
    """
    import numpy as np
    from scipy import stats
    
    print("Generating BMI distribution fit plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(bmi_data, kde=True, stat="density", color="skyblue", label="BMI Data")
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2, label="Normal Fit")
    
    plt.title("BMI Distribution vs. Normal Fit")
    plt.legend()
    plt.savefig('images/bmi_distribution_fit.png')
    plt.close()
    print("Saved 'images/bmi_distribution_fit.png'")

def plot_totchol_ci(data, ci_low, ci_high, mean):
    """
    Plots the density of totChol with 95% CI lines.
    Saves as 'images/totchol_ci_plot.png'.
    """
    print("Generating totChol CI plot...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, shade=True, color="purple", label="totChol Density")
    plt.axvline(ci_low, color='red', linestyle='--', label='Lower CI (95%)')
    plt.axvline(ci_high, color='green', linestyle='--', label='Upper CI (95%)')
    plt.axvline(mean, color='black', linestyle='-', label='Mean')
    plt.title(f"Density of totChol with 95% CI for Mean: [{ci_low:.2f}, {ci_high:.2f}]")
    plt.legend()
    plt.savefig('images/totchol_ci_plot.png')
    plt.close()
    print("Saved 'images/totchol_ci_plot.png'")

def plot_residuals(y_pred, residuals):
    """
    Plots residuals vs fitted values.
    Saves as 'images/residuals_plot.png'.
    """
    print("Generating residuals plot...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted Values (Predicted sysBP)')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted Values (Homoscedasticity Check)')
    plt.savefig('images/residuals_plot.png')
    plt.close()
    print("Saved 'images/residuals_plot.png'")
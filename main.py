import pandas as pd
from src import data_cleaning, visualization, inference

def main():
    # 1. Load and Clean Data
    filepath = 'data/framingham.csv' 
    # Note: Ensure you have a 'data' folder and the csv inside it, 
    # or adjust path to 'framingham.csv' if in root.
    
    df = data_cleaning.load_data(filepath)
    if df is None:
        return

    df_clean = data_cleaning.clean_data(df)

    # 2. Visualizations
    visualization.set_style()
    visualization.plot_distributions(df_clean)
    visualization.plot_relationships(df_clean)
    visualization.plot_multivariate(df_clean)
    visualization.plot_correlation(df_clean)

    # 3. Estimation & Inference
    # Get data back for specific plots
    bmi_data, totChol_data, ci_low, ci_high = inference.perform_estimations(df_clean)
    
    # Plot Estimation results
    visualization.plot_bmi_fit(bmi_data, bmi_data.mean(), bmi_data.std())
    visualization.plot_totchol_ci(totChol_data, ci_low, ci_high, totChol_data.mean())

    # Run Hypothesis Tests
    inference.hypothesis_testing(df_clean)

    # 4. Regression Analysis
    y_pred_lin, residuals = inference.run_regression_models(df_clean)
    
    # Plot Residuals
    visualization.plot_residuals(y_pred_lin, residuals)

    print("\nAnalysis Complete. All outputs saved to 'images/' directory.")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def perform_estimations(df):
    """
    Performs point estimation for sysBP, distribution fitting for BMI,
    and interval estimation for totChol.
    Returns data needed for plotting.
    """
    print("\n--- Starting Statistical Estimation ---")
    
    # MLE for sysBP
    mle_mean_sysBP = df['sysBP'].mean()
    mle_var_sysBP = df['sysBP'].var(ddof=0)
    print(f"MLE for sysBP Mean: {mle_mean_sysBP:.4f}")
    print(f"MLE for sysBP Variance: {mle_var_sysBP:.4f}")

    # BMI Normality Test
    bmi_clean = df['BMI'].dropna()
    shapiro_stat, shapiro_p = stats.shapiro(bmi_clean)
    print(f"Shapiro-Wilk Test for BMI: Statistic={shapiro_stat:.4f}, P-value={shapiro_p:.4g}")
    
    # totChol Confidence Interval
    totChol_clean = df['totChol'].dropna()
    ci_low, ci_high = stats.t.interval(0.95, len(totChol_clean)-1, loc=totChol_clean.mean(), scale=stats.sem(totChol_clean))
    print(f"95% Confidence Interval for mean totChol: ({ci_low:.4f}, {ci_high:.4f})")
    
    return bmi_clean, totChol_clean, ci_low, ci_high

def hypothesis_testing(df):
    """
    Performs T-test, Power Analysis, Mann-Whitney U test, and ANOVA.
    """
    print("\n--- Starting Hypothesis Testing ---")

    # 1. T-Test: Smokers vs. Non-Smokers (heartRate)
    smokers_hr = df[df['currentSmoker'] == 1]['heartRate'].dropna()
    nonsmokers_hr = df[df['currentSmoker'] == 0]['heartRate'].dropna()
    t_stat, p_val_ttest = stats.ttest_ind(smokers_hr, nonsmokers_hr, alternative='greater')
    print(f"T-Test (Smokers vs. Non-Smokers HR): T={t_stat:.4f}, P={p_val_ttest:.4g}")

    # 2. Power Analysis
    analysis = TTestIndPower()
    sample_size = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05, ratio=1.0, alternative='two-sided')
    print(f"Power Analysis (Effect=0.5, Power=0.8): Required Sample Size per Group = {sample_size:.2f}")

    # 3. Mann-Whitney U (Glucose by Diabetes)
    glucose_diabetes = df[df['diabetes'] == 1]['glucose'].dropna()
    glucose_no_diabetes = df[df['diabetes'] == 0]['glucose'].dropna()
    u_stat, p_val_mwu = stats.mannwhitneyu(glucose_diabetes, glucose_no_diabetes, alternative='two-sided')
    print(f"Mann-Whitney U (Glucose by Diabetes): U={u_stat:.4f}, P={p_val_mwu:.4g}")

    # 4. ANOVA (BMI across Education)
    anova_df = df[['BMI', 'education']].dropna()
    groups = [group['BMI'].values for name, group in anova_df.groupby('education')]
    f_stat, p_val_anova = stats.f_oneway(*groups)
    print(f"ANOVA (BMI across Education): F={f_stat:.4f}, P={p_val_anova:.4g}")
    
    if p_val_anova < 0.05:
        tukey = pairwise_tukeyhsd(endog=anova_df['BMI'], groups=anova_df['education'], alpha=0.05)
        print("Tukey's HSD Results:")
        print(tukey)

def run_regression_models(df):
    """
    Runs Logistic Regression for TenYearCHD, Lasso for sysBP, and checks Diagnostics.
    Returns prediction data for plotting residuals.
    """
    print("\n--- Starting Regression Analysis ---")

    # 1. Logistic Regression
    log_cols = ['age', 'male', 'sysBP', 'cigsPerDay', 'glucose', 'totChol']
    X_log = df[log_cols]
    y_log = df['TenYearCHD']
    
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.3, random_state=42)
    scaler_log = StandardScaler()
    X_train_log_scaled = scaler_log.fit_transform(X_train_log)
    X_test_log_scaled = scaler_log.transform(X_test_log)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train_log_scaled, y_train_log)
    y_pred_log = log_reg.predict(X_test_log_scaled)
    acc_log = accuracy_score(y_test_log, y_pred_log)
    
    print(f"Logistic Regression Accuracy: {acc_log:.4f}")
    print("Logistic Coefficients:")
    for feat, coef in zip(log_cols, log_reg.coef_[0]):
        print(f"  {feat}: {coef:.4f}")

    # 2. Lasso Regression (Predicting sysBP)
    lasso_features = ['age', 'cigsPerDay', 'totChol', 'diaBP', 'BMI', 'heartRate', 'glucose']
    X_lasso = df[lasso_features]
    y_lasso = df['sysBP']
    
    scaler_lasso = StandardScaler()
    X_lasso_scaled = scaler_lasso.fit_transform(X_lasso)
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_lasso_scaled, y_lasso)
    
    print("\nLasso Regression (sysBP) Coefficients:")
    for feat, coef in zip(lasso_features, lasso.coef_):
        if abs(coef) > 0.001:
            print(f"  {feat}: {coef:.4f}")

    # 3. Diagnostics: Linear Regression for Residuals & VIF
    X_lin = df[['BMI']] # Simplified for plot, or use full model
    y_lin = df['sysBP']
    lin_reg = LinearRegression()
    lin_reg.fit(X_lin, y_lin)
    y_pred_lin = lin_reg.predict(X_lin)
    residuals = y_lin - y_pred_lin
    
    # VIF Calculation
    X_vif = df[lasso_features].copy()
    X_vif['const'] = 1
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    print("\nVIF Results:")
    print(vif_data[vif_data['feature'] != 'const'])

    return y_pred_lin, residuals
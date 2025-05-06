import pandas as pd
import numpy as np
import utils as u
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages # Nice for summarized results

def print_null_zero_negative_counts(df):
    print("Null counts for each column:")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            nulls = df[col].isnull().sum()
            zeros = (df[col] == 0).sum()
            negatives = (df[col] < 0).sum()
        print(f"nulls: {nulls},  zeros: {zeros},  negatives:  {negatives} for column: {col}")

def histogram_and_quantiles(df, output_path="data/outputs/feature_hist_and_quartiles_charts.pdf", label_col="label"):
    with PdfPages(output_path) as pdf:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64'] and col != label_col:
                fig, axs = plt.subplots(nrows=4, figsize=(8, 6))
                fig.suptitle(f"{col} Histogram, Zeros, and 99th Quantile Volume", fontsize=14)

                plot_df = df[[col, label_col]].dropna()
                legit_values = plot_df[plot_df[label_col] == 0][col]
                phish_values = plot_df[plot_df[label_col] == 1][col]
                log_legit_values = np.log1p(plot_df[plot_df[label_col] == 0][col])
                log_phish_values = np.log1p(plot_df[plot_df[label_col] == 1][col])

                # Histogram with density=True
                axs[0].hist(legit_values, bins=50, alpha=0.6, label='Legit (0)', color='skyblue', density=True)
                axs[0].hist(phish_values, bins=50, alpha=0.6, label='Phishing (1)', color='salmon', density=True)
                axs[0].set_title("Histogram by Label")
                axs[0].legend()

                # Histogram with density=True
                axs[1].hist(log_legit_values, bins=50, alpha=0.6, label='Legit (0)', color='skyblue', density=True)
                axs[1].hist(log_phish_values, bins=50, alpha=0.6, label='Phishing (1)', color='salmon', density=True)
                axs[1].set_title("Log-Transformed Histogram by Label")
                axs[1].legend()

                # bar: % zero value bar chart
                zero_legit = (log_legit_values == 0).sum() / len(log_legit_values) * 100
                zero_phish = (log_phish_values == 0).sum() / len(log_phish_values) * 100
                axs[2].bar(['Legit (0)', 'Phishing (1)'], [zero_legit, zero_phish], color=['skyblue', 'salmon'])
                axs[2].set_title("% Zero Values by Label")
                axs[2].set_ylabel("% of rows")

                # bar: % of values above 99th quantile (overall) by label
                val_of_99_quantile = plot_df[col].quantile(0.99)
                sum_legit_99q = (legit_values > val_of_99_quantile).sum()
                sum_phish_99q = (phish_values > val_of_99_quantile).sum()
                legit_total = legit_values.sum()
                phish_total = phish_values.sum()
                legit_percent_above_99q = sum_legit_99q / legit_total
                phish_percent_above_99q = sum_phish_99q / phish_total
                axs[3].bar(['Legit (0)', 'Phishing (1)'], [legit_percent_above_99q, phish_percent_above_99q], color=['skyblue', 'salmon'])
                axs[3].set_title("% above 99th quantile by Label")
                axs[3].set_ylabel("% of rows")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

def sanity_checks(
        df,
        compare_cols = ['num_unique_words', 'num_stopwords', 'num_spelling_errors', 'num_urgent_keywords'], 
        base_col = 'num_words'
    ):
    '''num_words should not be shorter than [num_unique_words, num_stopwords, num_spelling_errors, num_urgent_keywords] '''
    compare_cols = compare_cols
    base_col = base_col
    violations = pd.Series(False, index=df.index)

    for col in compare_cols:
        this_violation = df[col] > df[base_col]
        num_violations = this_violation.sum()
        if num_violations > 0:
            print(f"{col}: {num_violations} rows where {col} > {base_col}")
        violations |= this_violation

    print(f"\nTotal rows with any violation: {violations.sum()} / {len(df)}")
    return violations

def run_explorations(df):
    columns = df.columns 
    shape = df.shape 
    print(df['label'].value_counts())
    print(f"size of the df is row: {shape[0]} and columns: {shape[1]}")
    print_null_zero_negative_counts(df)
    histogram_and_quantiles(df)
    sanity_checks(df)
    u.baseline_feature_importance(df)


if __name__ == "__main__":
    df = u.load_data()
    run_explorations(df)


    



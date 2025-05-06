import numpy as np
import utils as u

def add_ratio_features(df, ratio_specs = [
    ('num_unique_words', 'num_words', 'pct_unique_words'),
    ('num_links', 'num_words', 'links_per_word'),
    ('num_urgent_keywords', 'num_words', 'num_urgent_keywords_rate'),
    ('num_spelling_errors', 'num_words', 'spelling_error_rate'),
    ('num_stopwords', 'num_words', 'stopwords_rate'),
    ('num_unique_domains', 'num_links', 'num_unique_domains_rate'), # by num_links
]
):
    """
    Adds new columns to df that are ratios of two existing columns.

    Parameters:
    - df: pd.DataFrame
    - ratio_specs: list of tuples like (numerator_col, denominator_col, new_col_name)

    Returns:
    - df with new ratio columns added
    """
    for num_col, denom_col, new_col in ratio_specs:
        # Avoid division by zero with np.where
        df[new_col] = np.where(df[denom_col] != 0,
                               df[num_col] / df[denom_col],
                               0)
    return df

if __name__ == "__main__":
    df = u.load_data()
    u.baseline_feature_importance(df, "feature_importance_before_ratios.csv")
    df = add_ratio_features(df)
    u.baseline_feature_importance(df, "feature_importance_after_ratios.csv")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier # just for a quick baseline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

def load_data(location="data/email_phishing_data.csv"):
    print(f"retrieving data from {location}")
    df = pd.read_csv(location)
    if df.empty:
        print(f"The df loaded from {location} is empyt")
    return df

def baseline_feature_importance(df, output_filename="quick_feature_importance.csv"):
    output_location = "data/outputs/" + output_filename
    X = df.drop(columns='label')
    y = df['label']


    model = RandomForestClassifier().fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.plot(kind='barh')

    importances.to_csv(output_location, header=True)
    print(f"feature importances saved at {output_location}")

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
    """
    for num_col, denom_col, new_col in ratio_specs:
        # Avoid division by zero with np.where
        df[new_col] = np.where(df[denom_col] != 0,
                               df[num_col] / df[denom_col],
                               0)
    return df

def get_train_and_test_from_df(df, label="label", random_state=111, test_size=0.2):
    label = label
    X = df.drop(columns='label')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    return X_train, X_test, y_train, y_test





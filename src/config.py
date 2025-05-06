TEST_SIZE=0.2
RANDOM_STATE_TEST=111
RANDOM_STATE_VALIDATION=222
RATIO_SPECS = [
    ('num_unique_words', 'num_words', 'pct_unique_words'),
    ('num_links', 'num_words', 'links_per_word'),
    ('num_urgent_keywords', 'num_words', 'num_urgent_keywords_rate'),
    ('num_spelling_errors', 'num_words', 'spelling_error_rate'),
    ('num_stopwords', 'num_words', 'stopwords_rate'),
    ('num_unique_domains', 'num_links', 'num_unique_domains_rate'), # by num_links
]
DATA_INPUT_LOCATION = "data/email_phishing_data.csv"
LABEL = "label"

# GBM hyper parameters
GBM_RESULTS_LOCATION = 'data/outputs/gbm_grid_results.csv'
GBM_WEIGHTED_RESULTS_LOCATION = 'data/outputs/gbm_grid_results.csv'
GBM_PARAM_GRID_OLD = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
    "subsample": [0.8, 1.0]
}
GBM_PARAM_GRID = {
    "n_estimators": [300, 600, 900],
    "learning_rate": [0.1],
    "max_depth": [5],
    "min_samples_split": [5],
    "subsample": [0.8]
}
GBM_SELECTED_HYPERPARAMS = {
    "n_estimators": 900,
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_samples_split": 5,
    "subsample": 0.8
}
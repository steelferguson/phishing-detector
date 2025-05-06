import pandas as pd
import joblib
from feature_engineering import add_ratio_features
from config import DATA_INPUT_LOCATION
from sklearn.model_selection import train_test_split

# Load model and data
model = joblib.load('src/model_store/final_model.pkl')
df = pd.read_csv(DATA_INPUT_LOCATION)
df = add_ratio_features(df)

# Drop label and split
X = df.drop(columns='label')
y = df['label']

# Predict probabilities
probs = model.predict_proba(X)[:, 1]
df['probability'] = probs
df['label'] = y

# Filter high-confidence phishing
phish_high = df[(df['label'] == 1) & (df['probability'] > 0.9)]

# Show a few
print("High-confidence phishing samples:")
for _, row in phish_high.head(5).iterrows():
    payload = {k: int(row[k]) for k in [
        'num_words', 'num_unique_words', 'num_stopwords', 'num_links',
        'num_unique_domains', 'num_email_addresses', 'num_spelling_errors',
        'num_urgent_keywords'
    ]}
    print(payload)
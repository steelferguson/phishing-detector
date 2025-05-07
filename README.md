# phishing-detector
May 2025
Summary of this project
-  Use a scrubbed dataset with 9 features to create a phishing detector.
-  Initial exploratory data analysis with some visuals and statistic.
-  Load and train multiple types of models, evaluating each one.
-  Package the best model of the group in a REST API that accepts a JSON payload and returns a prediction.

# How to use the model

### Clone Repository
- git clone https://github.com/your-username/phishing-detector.git
- cd phishing-detector

### Create a virtual environment with the neede packages
- python -m venv venv
- source venv/bin/activate   # for mac

### Install required dependencies in the environment
- pip install -r requirements.txt

### Run the API locally
- uvicorn src.api.main:app --reload

### View the UI
- Navigate to http://127.0.0.1:8000/docs 


### Sample request
{
  "num_words": 100,
  "num_unique_words": 45,
  "num_stopwords": 30,
  "num_links": 2,
  "num_unique_domains": 1,
  "num_email_addresses": 0,
  "num_spelling_errors": 3,
  "num_urgent_keywords": 1
}

### Sample response
{
  "prediction": 1,
  "probability": 0.8432
}
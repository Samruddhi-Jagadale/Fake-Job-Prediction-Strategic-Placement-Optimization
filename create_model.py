import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load your dataset
fake_job_postings = pd.read_csv('fake_job_postings_cleaned.csv')

# Prepare the data
X = fake_job_postings['text']
y = fake_job_postings['fraudulent']  # Assuming this is your target column

# Load the vectorizer
count_vectorizer = joblib.load('vectorizer.pkl')
X_vectorized = count_vectorizer.transform(X)

# Train the Naive Bayes model
model = MultinomialNB(alpha=0.1)
model.fit(X_vectorized, y)

# Save the model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")

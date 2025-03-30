#pip install pandas numpy sklearn nltk

import pandas as pd
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("news.csv")  # Load dataset
print(df.head())  # Display first few rows

nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to clean text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()  # Tokenization
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

df["cleaned_text"] = df["text"].apply(clean_text)  # Apply cleaning


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_text"]).toarray()  # Convert text to numbers
y = df["label"].values  # Labels


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


news = ["Breaking: Scientists discover a new planet!"]
news_vectorized = vectorizer.transform(news).toarray()
prediction = model.predict(news_vectorized)
print("Fake News" if prediction[0] == 1 else "Real News")


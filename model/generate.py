import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import linear_kernel

np.random.seed(42)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def preprocess_text(text):
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    
    words = nltk.word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    words = [lemmatizer.lemmatize(word) for word in words]
    
    stop_words = set(stopwords.words('english'))
    
    words = [word for word in words if word not in stop_words]
    
    text = ' '.join(words).strip()
    
    return text


def train_model(df):
  
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df['preprocessed_questions'])

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, df['tag'], test_size=0.2, random_state=42)

    # Training a Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate accuracy and print the classification report
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    
    return model, tfidf_vectorizer, X


def generate_response(question, tfidf_vectorizer, tfidf_matrix, data):
  
    preprocessed_question = preprocess_text(question)
  
    question_vector = tfidf_vectorizer.transform([preprocessed_question])

    # Calculate cosine similarities between the input question and all questions in the dataset
    cosine_similarities = linear_kernel(question_vector, tfidf_matrix).flatten()

    # Find the index of the most similar question
    most_similar_index = np.argmax(cosine_similarities)

    # Return the answer of the most similar question
    return data.iloc[most_similar_index]['answer']


# Modify the interaction loop to use the updated function
while True:
    user_question = input("Ask a question (or type 'quit' to exit): ")
    if user_question.lower() == 'quit':
        break

    # Updated function call, bypassing tag predictions
    answer = generate_response(user_question, tfidf_vectorizer, X, df)

    print("Answer:", answer)
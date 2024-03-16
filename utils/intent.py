import re
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """
    Clean the input text by removing special characters and extra whitespace.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(file_path, max_words=10000):
    """
    Load data from an Excel file, preprocess it, and return the vectorized features and labels.

    Args:
        file_path (str): Path to the Excel file.
        max_words (int): Maximum number of words to keep during vectorization.

    Returns:
        tuple: A tuple containing vectorized features and labels.
    """
    data = pd.read_excel(file_path)
    texts = data['Text'].astype(str)
    labels = data['Class']

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    texts = texts.apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=max_words, stop_words='english')
    X_tfidf = vectorizer.fit_transform(texts)

    return X_tfidf, labels

def train_classifier(features, labels):
    """
    Train a Multinomial Naive Bayes classifier and return the trained classifier.

    Args:
        features: Vectorized text features.
        labels: Encoded labels.

    Returns:
        MultinomialNB: Trained classifier.
    """
    classifier = MultinomialNB()
    classifier.fit(features, labels)
    return classifier

def check_inference(text, classifier, vectorizer, label_encoder):
    """
    Perform inference on the input text using the provided classifier and return the predicted class.

    Args:
        text (str): Input text.
        classifier: Trained classifier.
        vectorizer: TF-IDF vectorizer.
        label_encoder: Label encoder.

    Returns:
        str: Predicted class.
    """
    cleaned_text = clean_text(text)
    tfidf = vectorizer.transform([cleaned_text])
    prediction = classifier.predict(tfidf)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    return predicted_class

if __name__ == "__main__":
    # Load and preprocess data
    X_tfidf, labels = load_and_preprocess_data('Dataset.xlsx')

    # Train the classifier
    classifier = train_classifier(X_tfidf, labels)

    # Example usage
    example_text = "This is an example text to classify."
    predicted_class = check_inference(example_text, classifier, vectorizer, label_encoder)
    print(f"Predicted class: {predicted_class}")

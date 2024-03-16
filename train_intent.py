import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pickle
from data_processing import load_and_preprocess_intent_dataset, clean_text

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and save an intent classification model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode.")
    return parser.parse_args()

def main(args):
    data_path = args.data_path
    verbose = args.verbose  # Check if verbose mode is enabled
    
    X_tfidf, labels, label_encoder, vectorizer = load_and_preprocess_intent_dataset(data_path)
    
    if verbose:
        print("Data loaded and preprocessed successfully.")
    
    classifier = MultinomialNB()
    classifier.fit(X_tfidf, labels)
    
    if verbose:
        print("Classifier trained successfully.")
    
    out_path = 'models/intent_classifier.pkl'
    with open(out_path, 'wb') as model_file:
        pickle.dump((classifier, vectorizer, label_encoder), model_file)
    
    if verbose:
        print(f"Model saved successfully at {out_path}\n")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
